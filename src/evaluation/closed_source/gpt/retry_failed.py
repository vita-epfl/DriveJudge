"""Retry the requests that failed in a previous batch run.

Two distinct failure modes to recover from:

* **Whole-batch rejection** (`batch.status == "failed"`): every request in the
  batch's JSONL never ran. We resubmit the full file.
* **Per-request failure inside a completed batch**: listed in the batch's
  `error_file`. We resubmit only those `custom_id`s, fished out of the
  original JSONL on disk.

Resubmission is throttled: **one batch in flight at a time**, so the total
enqueued tokens stay under OpenAI's per-model cap (2 M for `gpt-5.4-mini`).
That makes the whole run serial (~30 min per batch × N), but robust — the
script is resumable, so killing it and rerunning picks up where it stopped.

Sub-commands:

    build     Scan the existing state file + error files, produce
              `retry/jsonl/requests_NNN.jsonl` and `retry/retry_state.json`.
    submit    Submit the remaining retry parts, one at a time, waiting for
              each to finish. Idempotent — skips parts already submitted.
    peek      Download one error-file line verbatim, to see what an `unknown`
              error actually looks like in the wire format.

Typical flow:

    python retry_failed.py build
    python retry_failed.py peek                  # sanity-check the unknowns
    python retry_failed.py submit --wait         # long-running; resumable
    python retrieve_batch.py --state \\
        closed_source_evaluation/gpt/retry/retry_state.json
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

_HERE = Path(__file__).resolve().parent
DEFAULT_STATE = _HERE / "batches" / "batch_state.json"
DEFAULT_RETRY_DIR = _HERE / "retry"
# Same safety margin as the original run — stays well under the 200 MB file cap.
MAX_BYTES_PER_FILE = 150 * 1024 * 1024
# How often to re-poll when --wait is set.
POLL_INTERVAL_SECONDS = 120
_TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}


# ---------------------------------------------------------------------------
# Env + state helpers
# ---------------------------------------------------------------------------

def _load_env() -> None:
    for name in ("secrets.env", ".env"):
        p = _HERE / name
        if p.is_file():
            load_dotenv(p, override=False)
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY missing — see example.env.")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _download_error_file(client, file_id: str) -> list[dict]:
    raw = client.files.content(file_id).read().decode("utf-8")
    return [json.loads(l) for l in raw.splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# Build: collect lines that still need running and chunk them into new files
# ---------------------------------------------------------------------------

def cmd_build(args) -> None:
    state = _read_json(args.state)
    if "batches" not in state:
        sys.exit(f"{args.state} has no `batches` key — was it written by run_gpt.py?")

    _load_env()
    from openai import OpenAI
    client = OpenAI()

    retry_dir: Path = args.retry_dir
    jsonl_dir = retry_dir / "jsonl"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    for stale in jsonl_dir.glob("requests_*.jsonl"):
        stale.unlink()

    # Collect lines that still need to run, preserving their original custom_ids.
    retry_lines: list[tuple[str, str]] = []  # (custom_id, raw_jsonl_line_with_newline)
    summary = {"full_batch_retry": 0, "per_request_retry": 0, "skipped_completed": 0}

    for meta in state["batches"]:
        bid = meta["batch_id"]
        jsonl_path = Path(meta["jsonl_path"])
        if not jsonl_path.is_file():
            print(f"[warn] {jsonl_path} missing on disk; skipping batch {bid}")
            continue

        batch = client.batches.retrieve(bid)

        if batch.status == "failed":
            lines = _read_jsonl(jsonl_path)
            for obj in lines:
                # Re-serialise so whitespace is consistent and we can trust byte counts.
                line = json.dumps(obj, ensure_ascii=False) + "\n"
                retry_lines.append((obj["custom_id"], line))
            summary["full_batch_retry"] += len(lines)
            print(f"  {bid}  status=failed  → queuing all {len(lines)} requests for retry")
            continue

        if batch.status == "completed" and getattr(batch, "error_file_id", None):
            error_lines = _download_error_file(client, batch.error_file_id)
            failed_cids = {l["custom_id"] for l in error_lines if "custom_id" in l}
            if not failed_cids:
                print(f"  {bid}  status=completed  error_file present but no custom_ids (?) — skipping")
                continue
            originals_by_cid: dict[str, dict] = {}
            for obj in _read_jsonl(jsonl_path):
                if obj["custom_id"] in failed_cids:
                    originals_by_cid[obj["custom_id"]] = obj
            for cid in failed_cids:
                obj = originals_by_cid.get(cid)
                if obj is None:
                    print(f"    [warn] {cid} missing from {jsonl_path}")
                    continue
                line = json.dumps(obj, ensure_ascii=False) + "\n"
                retry_lines.append((cid, line))
            summary["per_request_retry"] += len(failed_cids)
            print(f"  {bid}  status=completed  → queuing {len(failed_cids)} failed requests for retry")
            continue

        # completed with no error_file, or in_progress/validating — nothing to retry here.
        summary["skipped_completed"] += 1

    if not retry_lines:
        print("\nNothing to retry. 🎉")
        return

    # Chunk to parts that respect the 150 MB cap.
    paths: list[Path] = []
    part_idx = -1
    fh = None
    cur_size = 0

    def _rotate() -> None:
        nonlocal part_idx, fh, cur_size
        if fh is not None:
            fh.close()
        part_idx += 1
        path = jsonl_dir / f"requests_{part_idx:03d}.jsonl"
        paths.append(path)
        fh = open(path, "w", encoding="utf-8")
        cur_size = 0

    _rotate()
    for _cid, line in retry_lines:
        line_bytes = len(line.encode("utf-8"))
        if cur_size + line_bytes > MAX_BYTES_PER_FILE and cur_size > 0:
            _rotate()
        assert fh is not None
        fh.write(line)
        cur_size += line_bytes
    if fh is not None:
        fh.close()

    # Load the mapping from the original state file so downstream retrieval
    # can still look up (video, question) per custom_id.
    original_mapping = state.get("mapping", {})

    retry_state = {
        "model": state.get("model"),
        "image_detail": state.get("image_detail"),
        "source_state": str(args.state),
        "parts": [
            {"jsonl_path": str(p),
             "size_bytes": p.stat().st_size,
             "num_requests": sum(1 for _ in open(p, encoding="utf-8"))}
            for p in paths
        ],
        "batches": [],     # populated incrementally by `submit`
        "mapping": original_mapping,
        "summary": summary,
    }
    _write_json(retry_dir / "retry_state.json", retry_state)

    total = len(retry_lines)
    print()
    print(f"Wrote {total} retry request(s) across {len(paths)} part(s):")
    for p in paths:
        print(f"  {p.name}  ({p.stat().st_size / 1e6:.1f} MB)")
    print(f"Summary: {summary}")
    print(f"\nNext:")
    print(f"  python retry_failed.py submit --wait")
    print(f"Then merge the retry results into a single JSON with:")
    print(f"  python retrieve_batch.py --state {retry_dir / 'retry_state.json'}")


# ---------------------------------------------------------------------------
# Submit: one at a time, waiting for each to finish before the next
# ---------------------------------------------------------------------------

def _wait_for_terminal(client, batch_id: str, *, poll: int) -> str:
    """Block until the batch reaches a terminal status; return that status."""
    while True:
        batch = client.batches.retrieve(batch_id)
        rc = getattr(batch, "request_counts", None)
        total = getattr(rc, "total", "?")
        completed = getattr(rc, "completed", "?")
        failed = getattr(rc, "failed", "?")
        print(f"    [{time.strftime('%H:%M:%S')}] {batch_id}  "
              f"status={batch.status}  completed={completed}/{total}  failed={failed}")
        if batch.status in _TERMINAL_STATUSES:
            return batch.status
        time.sleep(poll)


def cmd_submit(args) -> None:
    _load_env()
    from openai import OpenAI
    client = OpenAI()

    retry_state_path = args.retry_dir / "retry_state.json"
    if not retry_state_path.is_file():
        sys.exit(f"{retry_state_path} missing. Run `retry_failed.py build` first.")

    state = _read_json(retry_state_path)
    parts: list[dict] = state["parts"]
    submitted_paths = {b["jsonl_path"] for b in state["batches"]}
    pending = [p for p in parts if p["jsonl_path"] not in submitted_paths]
    if not pending:
        print("All retry parts already submitted. Use retrieve_batch.py to collect results.")
        return

    print(f"{len(parts)} part(s) total, {len(submitted_paths)} already submitted, "
          f"{len(pending)} pending.")
    model = state["model"]
    image_detail = state.get("image_detail")
    for i, part in enumerate(pending, start=1):
        jsonl_path = Path(part["jsonl_path"])
        print(f"\n[{i}/{len(pending)}] submitting {jsonl_path.name} "
              f"({part['num_requests']} requests, {part['size_bytes'] / 1e6:.1f} MB)…")
        up = client.files.create(file=open(jsonl_path, "rb"), purpose="batch")
        batch = client.batches.create(
            input_file_id=up.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"project": "DriveJudgeBench", "model": model,
                      "detail": image_detail, "part": jsonl_path.name, "retry": "1"},
        )
        print(f"    batch_id={batch.id}  status={batch.status}")
        record = {
            "batch_id": batch.id,
            "input_file_id": up.id,
            "status_at_submit": batch.status,
            "created_at": batch.created_at,
            "jsonl_path": str(jsonl_path),
        }
        state["batches"].append(record)
        _write_json(retry_state_path, state)

        if not args.wait:
            print("    --wait not set; exiting. Re-run to submit the next part "
                  "once this one completes.")
            return

        terminal = _wait_for_terminal(client, batch.id, poll=args.poll_interval)
        if terminal != "completed":
            print(f"    [warn] batch ended with status={terminal}. "
                  f"Continuing; inspect with diagnose_batches.py.")

    print("\nAll pending parts submitted.")


# ---------------------------------------------------------------------------
# Peek: show one raw line from an error_file so we can see the `unknown` errors
# ---------------------------------------------------------------------------

def cmd_peek(args) -> None:
    _load_env()
    from openai import OpenAI
    client = OpenAI()

    state = _read_json(args.state)
    if "batches" not in state:
        sys.exit(f"{args.state} has no `batches` key.")

    shown = 0
    for meta in state["batches"]:
        if shown >= args.num:
            break
        batch = client.batches.retrieve(meta["batch_id"])
        if batch.status != "completed" or not getattr(batch, "error_file_id", None):
            continue
        lines = _download_error_file(client, batch.error_file_id)
        for line in lines:
            if shown >= args.num:
                break
            print(f"--- {meta['batch_id']}  custom_id={line.get('custom_id')} ---")
            print(json.dumps(line, indent=2, ensure_ascii=False)[:2000])
            print()
            shown += 1

    if shown == 0:
        print("No error_file lines found in any completed batch.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--state", type=Path, default=DEFAULT_STATE,
                    help="Original batch state file (from run_gpt.py --mode batch).")
    ap.add_argument("--retry-dir", type=Path, default=DEFAULT_RETRY_DIR,
                    help="Where to write retry JSONLs + retry_state.json.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("build", help="Emit retry JSONLs from failed batches + error files.")

    p_submit = sub.add_parser("submit", help="Submit the next retry batch(es).")
    p_submit.add_argument("--wait", action="store_true",
                          help="Keep going until every retry part is submitted. "
                               "Without it, submits exactly one part and exits.")
    p_submit.add_argument("--poll-interval", type=int, default=POLL_INTERVAL_SECONDS,
                          help="Seconds between status polls when --wait is set.")

    p_peek = sub.add_parser("peek", help="Dump the first few raw error_file entries.")
    p_peek.add_argument("--num", type=int, default=3)

    args = ap.parse_args()

    if args.cmd == "build":
        cmd_build(args)
    elif args.cmd == "submit":
        cmd_submit(args)
    elif args.cmd == "peek":
        cmd_peek(args)


if __name__ == "__main__":
    main()
