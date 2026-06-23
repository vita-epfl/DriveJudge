"""Run GPT evaluation — live (synchronous) or batch (async submit).

Modes
-----
live
    Synchronous Chat Completions with structured output. Good for the 10-Q
    pilot because you get results + cached-token usage in seconds.
batch
    Writes a JSONL in OpenAI's batch format, uploads it, creates a batch job,
    saves `batch_state.json` with the batch ID + custom_id→item mapping, and
    exits. Nothing keeps running on your machine — pick results up later with
    `retrieve_batch.py`.

Output item shape (matches `analysis/answer_analysis_agent.py` expectations):
    {
      "video":      <cluster-style path, kept so the strip-prefix code works>,
      "question":   <original question text>,
      "evaluation": <rationale string>,
      "answer":     <normalised final answer>,
      "history":    [],            # unused for GPT, kept for schema parity
      "latency_s":  float,
      "usage":      {"input_tokens": int, "cached_input_tokens": int,
                     "output_tokens": int},
      "model":      "gpt-5.4-mini",
      "image_detail": "low" | "high" | "auto",
      "custom_id":  <stable id; also used as key in batch mode>,
    }
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

from frame_extractor import extract_frames
from path_fix import QuestionItem, load_questions
from prompt_builder import build_messages
from schemas import schema_for

_HERE = Path(__file__).resolve().parent
DEFAULT_QUESTIONS = Path("./dataset/dataset_final/Questions_raw.json")
DEFAULT_MODEL = "gpt-5.4-mini"


def _load_env() -> None:
    # Accept both `.env` and `secrets.env`; .env wins if both exist.
    for name in ("secrets.env", ".env"):
        p = _HERE / name
        if p.is_file():
            load_dotenv(p, override=False)
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit(
            f"OPENAI_API_KEY missing — set it in {_HERE / '.env'} "
            f"or export it in your shell. See example.env."
        )


def _custom_id(item: QuestionItem, idx: int) -> str:
    # Must be unique and stable. We include idx because the same (video,
    # question) pair shouldn't recur but a belt-and-braces tag is cheap.
    return f"q{idx:06d}_{Path(item.relative_path).stem[:40]}"


def _iter_jobs(items: list[QuestionItem], image_detail: str):
    for idx, item in enumerate(items):
        frames = extract_frames(item.local_path)
        schema = schema_for(item.question)
        messages = build_messages(frames, item.question, schema, image_detail=image_detail)
        yield idx, item, schema, messages


# ---------------------------------------------------------------------------
# Live mode
# ---------------------------------------------------------------------------

def run_live(
    items: list[QuestionItem],
    *,
    model: str,
    image_detail: str,
    output_path: Path,
) -> None:
    from openai import OpenAI

    client = OpenAI()
    results: list[dict[str, Any]] = []

    for idx, item, schema, messages in _iter_jobs(items, image_detail):
        t0 = time.perf_counter()
        resp = client.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=schema,
        )
        dt = time.perf_counter() - t0
        parsed = resp.choices[0].message.parsed
        usage = resp.usage
        cached = getattr(getattr(usage, "prompt_tokens_details", None),
                         "cached_tokens", 0) or 0
        out = {
            "video": item.cluster_path,
            "question": item.question,
            "evaluation": parsed.evaluation,
            "answer": parsed.answer,
            "history": [],
            "latency_s": round(dt, 3),
            "usage": {
                "input_tokens": usage.prompt_tokens,
                "cached_input_tokens": cached,
                "output_tokens": usage.completion_tokens,
            },
            "model": model,
            "image_detail": image_detail,
            "custom_id": _custom_id(item, idx),
        }
        results.append(out)
        print(
            f"[{idx + 1:>4}/{len(items)}] {dt:5.2f}s  "
            f"in={usage.prompt_tokens} (cached={cached})  "
            f"out={usage.completion_tokens}  → {parsed.answer}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(results)} results → {output_path}")


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------

def _batch_request(custom_id: str, model: str, messages: list[dict], schema_cls) -> dict:
    # OpenAI batch JSONL line for chat.completions with JSON-schema structured output.
    # Strict mode requires `additionalProperties: false` on every object — pydantic
    # doesn't emit it by default, so we add it ourselves.
    schema_json = schema_cls.model_json_schema()
    schema_json["additionalProperties"] = False
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_cls.__name__,
                    "strict": True,
                    "schema": schema_json,
                },
            },
        },
    }


# OpenAI caps a single batch input file at 200 MB; we flush well below that to
# stay safe across slight JSON-overhead fluctuations.
MAX_BYTES_PER_BATCH_FILE = 150 * 1024 * 1024


def _write_jsonl_chunks(
    items: list[QuestionItem],
    *,
    image_detail: str,
    model: str,
    jsonl_dir: Path,
    max_bytes_per_file: int,
) -> tuple[list[Path], dict[str, dict[str, Any]]]:
    """Serialise requests to `requests_000.jsonl`, `requests_001.jsonl`, …

    Rotates to a new file as soon as the current one would exceed
    `max_bytes_per_file`. Returns the ordered list of files written plus the
    `custom_id → metadata` mapping used by `retrieve_batch.py`.
    """
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    # Fresh start: remove any old parts so stale files don't get re-uploaded.
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
    try:
        for idx, item, schema, messages in _iter_jobs(items, image_detail):
            cid = _custom_id(item, idx)
            req = _batch_request(cid, model, messages, schema)
            line = json.dumps(req, ensure_ascii=False) + "\n"
            line_bytes = len(line.encode("utf-8"))

            if line_bytes > max_bytes_per_file:
                raise RuntimeError(
                    f"Single request {cid} is {line_bytes / 1e6:.1f} MB, "
                    f"larger than the {max_bytes_per_file / 1e6:.0f} MB file cap. "
                    f"Switch to `--image-detail low` or reduce --num-frames."
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
                "image_detail": image_detail,
            }
    finally:
        if fh is not None:
            fh.close()

    # If the last rotation produced an empty file (shouldn't happen given the
    # `cur_size > 0` guard, but be defensive), drop it.
    if paths and paths[-1].stat().st_size == 0:
        paths[-1].unlink()
        paths.pop()

    return paths, mapping


def run_batch_submit(
    items: list[QuestionItem],
    *,
    model: str,
    image_detail: str,
    state_path: Path,
    jsonl_dir: Path,
    max_bytes_per_file: int = MAX_BYTES_PER_BATCH_FILE,
) -> None:
    from openai import OpenAI

    client = OpenAI()
    paths, mapping = _write_jsonl_chunks(
        items,
        image_detail=image_detail,
        model=model,
        jsonl_dir=jsonl_dir,
        max_bytes_per_file=max_bytes_per_file,
    )
    print(f"Wrote {len(mapping)} requests across {len(paths)} JSONL part(s):")
    for p in paths:
        print(f"  {p.name}  ({p.stat().st_size / 1e6:.1f} MB)")

    batches: list[dict[str, Any]] = []
    for p in paths:
        up = client.files.create(file=open(p, "rb"), purpose="batch")
        batch = client.batches.create(
            input_file_id=up.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"project": "DriveJudgeBench", "model": model,
                      "detail": image_detail, "part": p.name},
        )
        batches.append({
            "batch_id": batch.id,
            "input_file_id": up.id,
            "status_at_submit": batch.status,
            "created_at": batch.created_at,
            "jsonl_path": str(p),
        })
        print(f"  submitted {p.name} → batch_id={batch.id}  status={batch.status}")

    state = {
        "model": model,
        "image_detail": image_detail,
        "batches": batches,
        "mapping": mapping,
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

    print(f"\n{len(batches)} batch(es) submitted; state file: {state_path}")
    print(f"Run `python retrieve_batch.py --state {state_path}` later to fetch results.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=("live", "batch"), required=True)
    ap.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N questions (useful for pilots).")
    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL))
    ap.add_argument("--image-detail", choices=("low", "high", "auto"), default="low")
    ap.add_argument("--output", type=Path,
                    default=_HERE / "results" / "gpt_live_results.json",
                    help="[live mode only] Where to save the results JSON.")
    ap.add_argument("--state", type=Path,
                    default=_HERE / "batches" / "batch_state.json",
                    help="[batch mode only] Where to save the submission state.")
    ap.add_argument("--jsonl-dir", type=Path,
                    default=_HERE / "batches" / "jsonl",
                    help="[batch mode only] Directory for batch input JSONL parts "
                         "(auto-chunked to stay under OpenAI's 200 MB file limit).")
    ap.add_argument("--max-bytes-per-file", type=int,
                    default=MAX_BYTES_PER_BATCH_FILE,
                    help="[batch mode only] Per-part JSONL size cap in bytes.")
    args = ap.parse_args()

    _load_env()
    items = load_questions(args.questions)
    if args.limit:
        items = items[: args.limit]
    print(f"Processing {len(items)} (video, question) pairs "
          f"with model={args.model}, detail={args.image_detail}.")

    if args.mode == "live":
        run_live(items, model=args.model, image_detail=args.image_detail,
                 output_path=args.output)
    else:
        run_batch_submit(items, model=args.model, image_detail=args.image_detail,
                         state_path=args.state, jsonl_dir=args.jsonl_dir,
                         max_bytes_per_file=args.max_bytes_per_file)


if __name__ == "__main__":
    main()
