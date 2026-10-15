"""Fetch Gemini batch result(s) and write them in the gpt-result JSON shape.

Default behaviour is **append-only**: if `--output` already exists, new items
are merged in by `custom_id` (latest line wins, like `merge_results.py` in
the gpt flow), so reruns add to the file without erasing past answers.

Usage:
    python retrieve_batch.py --state batches/batch_state.json
    python retrieve_batch.py --state batches/batch_state.json --wait
    python retrieve_batch.py --state batches/batch_state.json \\
        --output results/gemini_all_results.json
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

from gpt_helpers import output_stem

_HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = "gemini-3-flash-preview"
DEFAULT_STEM = "gemini"

_TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


def _load_env() -> None:
    for name in ("secrets.env", ".env"):
        p = _HERE / name
        if p.is_file():
            load_dotenv(p, override=False)
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        sys.exit("GEMINI_API_KEY missing - see example.env.")


def _load_state(state_path: Path) -> dict[str, Any]:
    with open(state_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _state_name(batch) -> str:
    state = getattr(batch, "state", None)
    return getattr(state, "name", str(state))


def _print_status(i: int, n: int, batch) -> None:
    name = _state_name(batch)
    print(f"  [{i + 1}/{n}] {batch.name}  state={name}")


def _extract_text(response: Any) -> str:
    """Pull the text out of a GenerateContentResponse-like object."""
    text = getattr(response, "text", None)
    if text:
        return text
    candidates = getattr(response, "candidates", None) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        if content is None:
            continue
        for part in getattr(content, "parts", []) or []:
            t = getattr(part, "text", None)
            if t:
                return t
    return ""


def _extract_usage(response: Any) -> dict[str, int]:
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
    return {
        "input_tokens": getattr(meta, "prompt_token_count", 0) or 0,
        "cached_input_tokens": getattr(meta, "cached_content_token_count", 0) or 0,
        "output_tokens": getattr(meta, "candidates_token_count", 0) or 0,
    }


def _parse_jsonl_text(content: bytes | str) -> list[dict[str, Any]]:
    if isinstance(content, bytes):
        content = content.decode("utf-8")
    return [json.loads(line) for line in content.splitlines() if line.strip()]


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


def _result_from_jsonl_line(line: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    cid = line.get("key") or line.get("custom_id") or ""
    if "error" in line and line["error"]:
        return _attach_reflection({
            "video": meta.get("video"),
            "question": meta.get("question"),
            "evaluation": "",
            "answer": "",
            "history": [],
            "error": line["error"],
            "custom_id": cid,
            "model": meta.get("model"),
        }, meta)

    response = line.get("response") or {}
    # Pull text from candidates -> content -> parts -> text.
    text = ""
    for cand in response.get("candidates", []) or []:
        for part in (cand.get("content") or {}).get("parts", []) or []:
            t = part.get("text") or ""
            if t:
                text = t
                break
        if text:
            break

    try:
        parsed = json.loads(text) if text else {}
    except json.JSONDecodeError:
        parsed = {"evaluation": text, "answer": ""}

    usage_md = response.get("usageMetadata") or response.get("usage_metadata") or {}
    usage = {
        "input_tokens": usage_md.get("promptTokenCount") or usage_md.get("prompt_token_count") or 0,
        "cached_input_tokens": (
            usage_md.get("cachedContentTokenCount")
            or usage_md.get("cached_content_token_count")
            or 0
        ),
        "output_tokens": (
            usage_md.get("candidatesTokenCount")
            or usage_md.get("candidates_token_count")
            or 0
        ),
    }

    return _attach_reflection({
        "video": meta.get("video"),
        "question": meta.get("question"),
        "evaluation": parsed.get("evaluation", ""),
        "answer": parsed.get("answer", ""),
        "history": [],
        "usage": usage,
        "model": meta.get("model"),
        "custom_id": cid,
    }, meta)


def _result_from_inline(idx: int, inlined, mapping: dict[str, dict[str, Any]],
                        ordered_cids: list[str]) -> dict[str, Any]:
    cid = ordered_cids[idx] if idx < len(ordered_cids) else f"unknown_{idx}"
    meta = mapping.get(cid, {})
    err = getattr(inlined, "error", None)
    if err is not None:
        return _attach_reflection({
            "video": meta.get("video"),
            "question": meta.get("question"),
            "evaluation": "",
            "answer": "",
            "history": [],
            "error": str(err),
            "custom_id": cid,
            "model": meta.get("model"),
        }, meta)
    response = getattr(inlined, "response", None)
    text = _extract_text(response) if response else ""
    try:
        parsed = json.loads(text) if text else {}
    except json.JSONDecodeError:
        parsed = {"evaluation": text, "answer": ""}
    return _attach_reflection({
        "video": meta.get("video"),
        "question": meta.get("question"),
        "evaluation": parsed.get("evaluation", ""),
        "answer": parsed.get("answer", ""),
        "history": [],
        "usage": _extract_usage(response) if response else
                 {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0},
        "model": meta.get("model"),
        "custom_id": cid,
    }, meta)


def _collect_results_for_batch(client, batch, mapping: dict[str, dict[str, Any]]
                               ) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    dest = getattr(batch, "dest", None)
    if dest is None:
        return results

    file_name = getattr(dest, "file_name", None)
    if file_name:
        content = client.files.download(file=file_name)
        for line in _parse_jsonl_text(content):
            cid = line.get("key") or ""
            meta = mapping.get(cid, {})
            results.append(_result_from_jsonl_line(line, meta))
        return results

    inlined = getattr(dest, "inlined_responses", None)
    if inlined:
        # Inline batches return results in submit order; map back via batch
        # input file path -> JSONL keys (best-effort).
        ordered_cids: list[str] = []
        # We can't easily recover order from just the API response, so fall
        # back to mapping lookup by trying to match each response by custom_id
        # if present (the SDK exposes `key` on inline responses in newer
        # versions; older ones don't). When key is missing, we surface
        # whatever metadata we can.
        for i, item in enumerate(inlined):
            cid = getattr(item, "key", None) or ""
            meta = mapping.get(cid, {}) if cid else {}
            results.append(_result_from_inline(i, item, mapping, ordered_cids or [cid]))
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--state", type=Path,
                    default=_HERE / "batches" / "batch_state.json")
    ap.add_argument("--output", type=Path, default=None,
                    help="Result JSON path. Defaults to "
                         "results/<stem>_batch_results.json, where <stem> is "
                         "`gemini` for the default model and the sanitised "
                         "model id (read from the state file) otherwise. If it "
                         "exists, new results are merged in (latest wins per "
                         "custom_id).")
    ap.add_argument("--wait", action="store_true",
                    help="Poll every 60s until every batch reaches a terminal state.")
    args = ap.parse_args()

    _load_env()
    from google import genai
    client = genai.Client()

    state = _load_state(args.state)
    mapping: dict[str, dict[str, Any]] = state.get("mapping", {})
    batch_names: list[str] = [b["batch_name"] for b in state.get("batches", [])]

    if args.output is None:
        stem = output_stem(state.get("model", DEFAULT_MODEL),
                           default_model=DEFAULT_MODEL, default_stem=DEFAULT_STEM)
        suffix = "_r2" if any("answer_r1" in m for m in mapping.values()) else ""
        args.output = _HERE / "results" / f"{stem}_batch_results{suffix}.json"

    n = len(batch_names)
    print(f"Tracking {n} batch(es) from {args.state} -> {args.output}.")

    while True:
        print(f"[{time.strftime('%H:%M:%S')}] polling...")
        batches = [client.batches.get(name=bn) for bn in batch_names]
        for i, b in enumerate(batches):
            _print_status(i, n, b)
        if all(_state_name(b) in _TERMINAL_STATES for b in batches):
            break
        if not args.wait:
            print("\nNot all finished - rerun later, or pass --wait to poll.")
            return
        time.sleep(60)

    failed = [b for b in batches if _state_name(b) != "JOB_STATE_SUCCEEDED"]
    if failed:
        print(f"\n[warn] {len(failed)} batch(es) did not succeed:")
        for b in failed:
            print(f"  {b.name}  state={_state_name(b)}  error={getattr(b, 'error', None)}")

    new_results: list[dict[str, Any]] = []
    for b in batches:
        if _state_name(b) == "JOB_STATE_SUCCEEDED":
            new_results.extend(_collect_results_for_batch(client, b, mapping))

    # Append-style merge: existing file + new_results, dedup by custom_id,
    # latest (= newly fetched) wins.
    merged: dict[str, dict[str, Any]] = {}
    if args.output.is_file():
        with open(args.output, "r", encoding="utf-8") as f:
            for it in json.load(f):
                cid = it.get("custom_id")
                if cid:
                    merged[cid] = it
        print(f"Loaded {len(merged)} prior result(s) from {args.output}.")
    for it in new_results:
        cid = it.get("custom_id")
        if cid:
            merged[cid] = it

    final = list(merged.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    ok = sum(1 for r in final if r.get("answer") and not r.get("error"))
    err = sum(1 for r in final if r.get("error") or not r.get("answer"))
    print(f"\nFetched {len(new_results)} new result(s); merged total = {len(final)}.")
    print(f"  OK     : {ok}")
    print(f"  Errored: {err}")
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
