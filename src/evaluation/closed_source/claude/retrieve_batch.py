"""Fetch Claude Message Batches and write them in the shared result JSON shape.

Default behaviour is **append-only**: if `--output` already exists, new items
are merged in by `custom_id` (latest wins), so reruns add to the file without
erasing past answers — same contract as the gemini/ retriever.

Usage:
    python retrieve_batch.py --state batches/batch_state.json
    python retrieve_batch.py --state batches/batch_state.json --wait
    python retrieve_batch.py --state batches/batch_state.json \\
        --output results/claude_all_results.json
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

from claude_request import extract_usage, parse_result
from gpt_helpers import output_stem

_HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_STEM = "claude"


def _load_env() -> None:
    for name in ("secrets.env", ".env"):
        p = _HERE / name
        if p.is_file():
            load_dotenv(p, override=False)
    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        sys.exit("ANTHROPIC_API_KEY missing - see example.env.")


def _load_state(state_path: Path) -> dict[str, Any]:
    with open(state_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _base_record(cid: str, meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "video": meta.get("video"),
        "question": meta.get("question"),
        "evaluation": "",
        "answer": "",
        "history": [],
        "model": meta.get("model"),
        "thinking": meta.get("thinking"),
        "custom_id": cid,
    }


def _attach_reflection(rec: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    """Add the round-1 columns when this batch was a `--reflect` round."""
    if "answer_r1" not in meta:
        return rec
    rec["evaluation_r1"] = meta.get("evaluation_r1", "")
    rec["answer_r1"] = meta.get("answer_r1", "")
    rec["changed"] = rec.get("answer") != rec["answer_r1"]
    round1 = meta.get("round1")
    rec["history"] = [round1] if round1 else []
    return rec


def _result_to_record(entry: Any, mapping: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cid = entry.custom_id
    meta = mapping.get(cid, {})
    rec = _base_record(cid, meta)
    result = entry.result
    rtype = result.type

    if rtype != "succeeded":
        err = getattr(result, "error", None)
        rec["error"] = f"{rtype}: {err}" if err is not None else rtype
        return _attach_reflection(rec, meta)

    message = result.message
    try:
        parsed = parse_result(message)
    except Exception as exc:  # noqa: BLE001 — keep the row, flag the failure
        rec["error"] = f"parse_failed: {type(exc).__name__}: {exc}"
        rec["usage"] = extract_usage(message)
        return _attach_reflection(rec, meta)

    rec["evaluation"] = parsed["evaluation"]
    rec["answer"] = parsed["answer"]
    rec["usage"] = extract_usage(message)
    rec["stop_reason"] = getattr(message, "stop_reason", None)
    return _attach_reflection(rec, meta)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--state", type=Path,
                    default=_HERE / "batches" / "batch_state.json")
    ap.add_argument("--output", type=Path, default=None,
                    help="Result JSON path. Defaults to "
                         "results/<model-stem>_batch_results.json (the stem is "
                         "`claude` for the default model). Existing files are "
                         "merged into, latest wins per custom_id.")
    ap.add_argument("--wait", action="store_true",
                    help="Poll every 60s until every batch has ended.")
    ap.add_argument("--poll-seconds", type=int, default=60)
    args = ap.parse_args()

    _load_env()
    import anthropic
    client = anthropic.Anthropic()

    state = _load_state(args.state)
    mapping: dict[str, dict[str, Any]] = state.get("mapping", {})
    batch_ids: list[str] = [b["batch_id"] for b in state.get("batches", [])]
    if not batch_ids:
        sys.exit(f"{args.state} lists no batches.")

    if args.output is None:
        stem = output_stem(state.get("model", DEFAULT_MODEL),
                           default_model=DEFAULT_MODEL, default_stem=DEFAULT_STEM)
        suffix = "_r2" if any("answer_r1" in m for m in mapping.values()) else ""
        args.output = _HERE / "results" / f"{stem}_batch_results{suffix}.json"

    n = len(batch_ids)
    print(f"Tracking {n} batch(es) from {args.state} -> {args.output}")

    while True:
        print(f"[{time.strftime('%H:%M:%S')}] polling...")
        batches = [client.messages.batches.retrieve(bid) for bid in batch_ids]
        for i, b in enumerate(batches):
            rc = getattr(b, "request_counts", None)
            print(f"  [{i + 1}/{n}] {b.id}  status={b.processing_status}  "
                  f"succeeded={getattr(rc, 'succeeded', '?')} "
                  f"processing={getattr(rc, 'processing', '?')} "
                  f"errored={getattr(rc, 'errored', '?')}")
        if all(b.processing_status == "ended" for b in batches):
            break
        if not args.wait:
            print("\nNot all ended - rerun later, or pass --wait to poll.")
            return
        time.sleep(args.poll_seconds)

    new_results: list[dict[str, Any]] = []
    problems: dict[str, list[str]] = {}
    for b in batches:
        for entry in client.messages.batches.results(b.id):
            rec = _result_to_record(entry, mapping)
            new_results.append(rec)
            rtype = entry.result.type
            if rtype != "succeeded" or rec.get("error"):
                problems.setdefault(rtype, []).append(entry.custom_id)

    for rtype, cids in sorted(problems.items()):
        print(f"\n[warn] {len(cids)} request(s) {rtype}. First few: {cids[:5]}")

    merged: dict[str, dict[str, Any]] = {}
    if args.output.is_file():
        with open(args.output, "r", encoding="utf-8") as f:
            for it in json.load(f):
                if it.get("custom_id"):
                    merged[it["custom_id"]] = it
        print(f"Loaded {len(merged)} prior result(s) from {args.output}.")
    for it in new_results:
        merged[it["custom_id"]] = it

    final = list(merged.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    ok = sum(1 for r in final if r.get("answer") and not r.get("error"))
    err = len(final) - ok
    print(f"\nFetched {len(new_results)} new result(s); merged total = {len(final)}.")
    print(f"  OK     : {ok}")
    print(f"  Errored: {err}")
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
