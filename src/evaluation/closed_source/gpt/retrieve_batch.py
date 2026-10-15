"""Fetch previously-submitted batch(es) and materialise their results.

Large runs are split across multiple batches to stay under OpenAI's 200 MB
input-file limit. The state file from `run_gpt.py --mode batch` lists all of
them; this script retrieves each in turn and merges their outputs into a
single results JSON.

Usage:
    python retrieve_batch.py --state batches/batch_state.json
    python retrieve_batch.py --state batches/batch_state.json --wait

`--wait` polls every 60s until every batch is in a terminal state. Without
it, the script prints per-batch status and exits immediately — safe to rerun
any time (idempotent).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from naming import output_stem

_HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_STEM = "gpt"


def _load_env() -> None:
    for name in ("secrets.env", ".env"):
        p = _HERE / name
        if p.is_file():
            load_dotenv(p, override=False)
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY missing — see example.env.")


def _load_state(state_path: Path) -> dict[str, Any]:
    with open(state_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_output_file(client, output_file_id: str) -> list[dict[str, Any]]:
    content = client.files.content(output_file_id).read()
    lines = content.decode("utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _attach_reflection(rec: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    """Add the round-1 columns when the batch was a `--reflect` round."""
    if "answer_r1" not in meta:
        return rec
    rec["evaluation_r1"] = meta.get("evaluation_r1", "")
    rec["answer_r1"] = meta.get("answer_r1", "")
    rec["changed"] = rec.get("answer") != rec["answer_r1"]
    round1 = meta.get("round1")
    rec["history"] = [round1] if round1 else []
    return rec


def _to_result_item(
    line: dict[str, Any],
    mapping: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    cid = line["custom_id"]
    meta = mapping.get(cid, {})
    if line.get("error") or line.get("response", {}).get("status_code", 0) >= 400:
        return _attach_reflection({
            "video": meta.get("video"),
            "question": meta.get("question"),
            "evaluation": "",
            "answer": "",
            "history": [],
            "error": line.get("error") or line["response"].get("body"),
            "custom_id": cid,
            "model": meta.get("model"),
            "image_detail": meta.get("image_detail"),
        }, meta)
    body = line["response"]["body"]
    choice = body["choices"][0]["message"]
    # Structured output can land either as parsed JSON in `content` or as a
    # tool/function arguments string — the JSON-schema route uses `content`.
    content = choice.get("content") or ""
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {"evaluation": content, "answer": ""}

    usage = body.get("usage", {}) or {}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)

    return _attach_reflection({
        "video": meta.get("video"),
        "question": meta.get("question"),
        "evaluation": parsed.get("evaluation", ""),
        "answer": parsed.get("answer", ""),
        "history": [],
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "cached_input_tokens": cached,
            "output_tokens": usage.get("completion_tokens", 0),
        },
        "model": meta.get("model"),
        "image_detail": meta.get("image_detail"),
        "custom_id": cid,
    }, meta)


_TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}


def _print_status(i: int, n: int, batch) -> None:
    rc = getattr(batch, "request_counts", None)
    total = getattr(rc, "total", "?")
    completed = getattr(rc, "completed", "?")
    failed = getattr(rc, "failed", "?")
    print(f"  [{i + 1}/{n}] {batch.id}  status={batch.status}  "
          f"completed={completed}/{total}  failed={failed}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--state", type=Path,
                    default=_HERE / "batches" / "batch_state.json")
    ap.add_argument("--output", type=Path, default=None,
                    help="Result JSON path. Defaults to "
                         "results/<stem>_batch_results.json, where <stem> is "
                         "`gpt` for the default model and the sanitised model "
                         "id (read from the state file) otherwise.")
    ap.add_argument("--wait", action="store_true",
                    help="Poll every 60s until every batch reaches a terminal status.")
    args = ap.parse_args()

    _load_env()
    from openai import OpenAI
    client = OpenAI()

    state = _load_state(args.state)
    mapping: dict[str, Any] = state.get("mapping", {})

    if args.output is None:
        stem = output_stem(state.get("model", DEFAULT_MODEL),
                           default_model=DEFAULT_MODEL, default_stem=DEFAULT_STEM)
        suffix = "_r2" if any("answer_r1" in m for m in mapping.values()) else ""
        args.output = _HERE / "results" / f"{stem}_batch_results{suffix}.json"
    print(f"Writing results to {args.output}")

    # Accept both the current chunked schema (`state["batches"]` is a list)
    # and the pre-chunking schema (`state["batch_id"]` at the top level).
    if "batches" in state:
        batch_ids: list[str] = [b["batch_id"] for b in state["batches"]]
    elif "batch_id" in state:
        batch_ids = [state["batch_id"]]
    else:
        sys.exit(f"{args.state} has neither `batches` nor `batch_id` — "
                 f"can't tell what to retrieve.")
    n = len(batch_ids)
    print(f"Tracking {n} batch(es) from {args.state}.")

    while True:
        print(f"[{time.strftime('%H:%M:%S')}] polling…")
        batches = [client.batches.retrieve(bid) for bid in batch_ids]
        for i, b in enumerate(batches):
            _print_status(i, n, b)

        if all(b.status in _TERMINAL_STATUSES for b in batches):
            break
        if not args.wait:
            print("\nNot all finished — rerun later, or pass --wait to poll.")
            return
        time.sleep(60)

    failed_batches = [b for b in batches if b.status != "completed"]
    if failed_batches:
        print(f"\n[warn] {len(failed_batches)} batch(es) did not complete cleanly:")
        for b in failed_batches:
            print(f"  {b.id}  status={b.status}  error_file={b.error_file_id}")

    results: list[dict[str, Any]] = []
    for b in batches:
        if b.status == "completed" and b.output_file_id:
            for line in _parse_output_file(client, b.output_file_id):
                item = _to_result_item(line, mapping)
                if item is not None:
                    results.append(item)
        if b.error_file_id:
            err_lines = _parse_output_file(client, b.error_file_id)
            print(f"  {b.id}: {len(err_lines)} errored request(s) in error file.")
            for line in err_lines:
                item = _to_result_item(line, mapping)
                if item is not None:
                    results.append(item)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(results)} results → {args.output}")


if __name__ == "__main__":
    main()
