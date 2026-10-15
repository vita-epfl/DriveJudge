"""Submit a Gemini batch job for the VLM-eval dataset.

Mirrors `closed_source_evaluation/gpt/run_gpt.py --mode batch` but targets the
Gemini Batch API (file-based JSONL upload). Output JSON shape matches the gpt
flow so downstream `analysis/answer_analysis_agent.py` works unchanged.

Memory / append-only behaviour:
    --skip-existing PATH    Load an existing results JSON (e.g. previous
                            runs merged together) and skip every (video,
                            question) pair whose `custom_id` is already
                            present with a non-empty `answer`. Lets you
                            top-up runs without resubmitting completed work.

Typical flow:
    # 1. smoke test with 2 questions
    python run_gemini.py --limit 2

    # 2. retrieve when done
    python retrieve_batch.py --state batches/batch_state.json --wait

    # 3. later, run on everything else but skip the 2 we already have
    python run_gemini.py --skip-existing results/gemini_all_results.json
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

from gpt_helpers import (
    QuestionItem,
    extract_frames,
    load_questions,
    output_stem,
    schema_for,
)
from gemini_request import batch_jsonl_line, build_request_body

_HERE = Path(__file__).resolve().parent
DEFAULT_QUESTIONS = Path("./dataset/dataset_final/Questions_raw.json")
DEFAULT_MODEL = "gemini-3-flash-preview"
DEFAULT_STEM = "gemini"

# Gemini's batch input file cap is 2 GB; we stay well below that to avoid
# accidental rejects on borderline files.
MAX_BYTES_PER_BATCH_FILE = 1_500 * 1024 * 1024


def _load_env() -> None:
    for name in ("secrets.env", ".env"):
        p = _HERE / name
        if p.is_file():
            load_dotenv(p, override=False)
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        sys.exit(
            f"GEMINI_API_KEY missing - set it in {_HERE / '.env'} "
            f"or export it in your shell. See example.env."
        )


def _custom_id(item: QuestionItem, idx: int) -> str:
    return f"q{idx:06d}_{Path(item.relative_path).stem[:40]}"


def _load_done_custom_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    done: set[str] = set()
    for it in items:
        cid = it.get("custom_id")
        if not cid:
            continue
        # Only count successful answers as done; errored rows should retry.
        if it.get("answer") and not it.get("error"):
            done.add(cid)
    return done


def _load_round1(path: Path) -> dict[str, dict[str, Any]]:
    """custom_id -> round-1 record (successful answers only), for --reflect."""
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    return {
        it["custom_id"]: it
        for it in items
        if it.get("custom_id") and it.get("answer") and not it.get("error")
    }


def _write_jsonl_chunks(
    items: list[QuestionItem],
    *,
    skip_cids: set[str],
    jsonl_dir: Path,
    max_bytes_per_file: int,
    model: str,
    round1: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[Path], dict[str, dict[str, Any]]]:
    """Write batch input parts. Returns (paths, custom_id -> meta mapping)."""
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    for stale in jsonl_dir.glob("requests_*.jsonl"):
        stale.unlink()

    mapping: dict[str, dict[str, Any]] = {}
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
    skipped = 0
    no_round1 = 0
    try:
        for idx, item in enumerate(items):
            base_cid = _custom_id(item, idx)
            prev = None
            if round1 is not None:
                prev = round1.get(base_cid)
                if prev is None:
                    no_round1 += 1
                    continue
            cid = f"{base_cid}_r2" if round1 is not None else base_cid
            if cid in skip_cids:
                skipped += 1
                continue
            frames = extract_frames(item.local_path)
            schema = schema_for(item.question)
            body = build_request_body(frames, item.question, schema, reflection=prev)
            line_obj = batch_jsonl_line(cid, body)
            line = json.dumps(line_obj, ensure_ascii=False) + "\n"
            line_bytes = len(line.encode("utf-8"))

            if line_bytes > max_bytes_per_file:
                raise RuntimeError(
                    f"Single request {cid} is {line_bytes / 1e6:.1f} MB, "
                    f"larger than the {max_bytes_per_file / 1e6:.0f} MB file cap."
                )
            if cur_size + line_bytes > max_bytes_per_file and cur_size > 0:
                _rotate()

            assert fh is not None
            fh.write(line)
            cur_size += line_bytes
            mapping[cid] = {
                "video": item.cluster_path,
                "question": item.question,
                "schema": schema.__name__,
                "model": model,
            }
            if prev is not None:
                mapping[cid]["evaluation_r1"] = prev.get("evaluation", "")
                mapping[cid]["answer_r1"] = prev.get("answer", "")
                mapping[cid]["round1"] = prev
    finally:
        if fh is not None:
            fh.close()

    if paths and paths[-1].stat().st_size == 0:
        paths[-1].unlink()
        paths.pop()

    if skipped:
        print(f"Skipped {skipped} already-answered request(s) (--skip-existing).")
    if no_round1:
        print(f"Skipped {no_round1} item(s) with no successful round-1 answer.")
    return paths, mapping


def run_batch_submit(
    items: list[QuestionItem],
    *,
    model: str,
    state_path: Path,
    jsonl_dir: Path,
    skip_existing: Path | None,
    max_bytes_per_file: int = MAX_BYTES_PER_BATCH_FILE,
    round1: dict[str, dict[str, Any]] | None = None,
) -> None:
    from google import genai
    from google.genai import types

    client = genai.Client()

    skip_cids = _load_done_custom_ids(skip_existing) if skip_existing else set()
    if skip_cids:
        print(f"Loaded {len(skip_cids)} already-answered custom_ids from "
              f"{skip_existing}.")

    paths, mapping = _write_jsonl_chunks(
        items,
        skip_cids=skip_cids,
        jsonl_dir=jsonl_dir,
        max_bytes_per_file=max_bytes_per_file,
        model=model,
        round1=round1,
    )
    if not paths:
        print("Nothing new to submit.")
        return

    print(f"Wrote {len(mapping)} requests across {len(paths)} JSONL part(s):")
    for p in paths:
        print(f"  {p.name}  ({p.stat().st_size / 1e6:.1f} MB)")

    batches: list[dict[str, Any]] = []
    for p in paths:
        upload = client.files.upload(
            file=str(p),
            config=types.UploadFileConfig(
                display_name=f"vlm-eval-gemini-{int(time.time())}-{p.name}",
                mime_type="jsonl",
            ),
        )
        batch = client.batches.create(
            model=model,
            src=upload.name,
            config={"display_name": f"vlm-eval-gemini-{p.name}"},
        )
        batches.append({
            "batch_name": batch.name,
            "input_file": upload.name,
            "state_at_submit": getattr(batch.state, "name", str(batch.state)),
            "jsonl_path": str(p),
        })
        print(f"  submitted {p.name} -> batch={batch.name}  "
              f"state={getattr(batch.state, 'name', batch.state)}")

    state = {
        "model": model,
        "batches": batches,
        "mapping": mapping,
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    # Append-style: merge with any existing mapping so previous batches stay
    # tracked and `retrieve_batch.py` can still see their results.
    if state_path.is_file():
        with open(state_path, "r", encoding="utf-8") as f:
            old = json.load(f)
        old_mapping = old.get("mapping", {})
        old_mapping.update(mapping)
        state["mapping"] = old_mapping
        state["batches"] = (old.get("batches", []) + batches)
        print(f"Merged with existing state at {state_path} "
              f"({len(state['batches'])} batches total).")

    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

    print(f"\n{len(batches)} batch(es) submitted; state file: {state_path}")
    print(f"Run `python retrieve_batch.py --state {state_path}` later to fetch results.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N questions (useful for pilots).")
    ap.add_argument("--model", default=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL))
    ap.add_argument("--skip-existing", type=Path, default=None,
                    help="Path to a results JSON whose successful custom_ids "
                         "should be skipped.")
    ap.add_argument("--out-name", default=None,
                    help="Filename stem for the run's artifacts. Defaults to "
                         "`gemini` for the default model and to the sanitised "
                         "model id for any other, so two models never collide.")
    ap.add_argument("--reflect", action="store_true",
                    help="Round-2 self-reflection pass; requires --from.")
    ap.add_argument("--from", dest="from_results", type=Path, default=None,
                    help="[--reflect] Round-1 results JSON to re-examine.")
    ap.add_argument("--state", type=Path, default=None,
                    help="Where to save the submission state "
                         "(default batches/batch_state.json for the default model).")
    ap.add_argument("--jsonl-dir", type=Path, default=None,
                    help="Directory for batch input JSONL parts.")
    ap.add_argument("--max-bytes-per-file", type=int,
                    default=MAX_BYTES_PER_BATCH_FILE,
                    help="Per-part JSONL size cap in bytes.")
    args = ap.parse_args()

    if args.reflect and args.from_results is None:
        sys.exit("--reflect needs --from <round-1 results json>.")
    if args.from_results is not None and not args.reflect:
        sys.exit("--from only makes sense together with --reflect.")

    stem = args.out_name or output_stem(
        args.model, default_model=DEFAULT_MODEL, default_stem=DEFAULT_STEM)
    suffix = "_r2" if args.reflect else ""
    if args.state is None:
        name = "batch_state" if stem == DEFAULT_STEM else f"batch_state_{stem}"
        args.state = _HERE / "batches" / f"{name}{suffix}.json"
    if args.jsonl_dir is None:
        name = "jsonl" if stem == DEFAULT_STEM else f"jsonl_{stem}"
        args.jsonl_dir = _HERE / "batches" / f"{name}{suffix}"

    _load_env()
    items = load_questions(args.questions)
    if args.limit:
        items = items[: args.limit]
    round1 = _load_round1(args.from_results) if args.reflect else None
    if round1 is not None:
        print(f"Loaded {len(round1)} round-1 answer(s) from {args.from_results}.")
    print(f"Processing {len(items)} (video, question) pairs with model={args.model}"
          f"{', reflect' if args.reflect else ''}.")

    run_batch_submit(
        items,
        model=args.model,
        state_path=args.state,
        jsonl_dir=args.jsonl_dir,
        skip_existing=args.skip_existing,
        max_bytes_per_file=args.max_bytes_per_file,
        round1=round1,
    )


if __name__ == "__main__":
    main()
