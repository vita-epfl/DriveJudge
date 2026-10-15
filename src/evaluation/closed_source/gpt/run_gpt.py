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
from naming import output_stem
from path_fix import QuestionItem, load_questions
from prompt_builder import build_messages
from schemas import schema_for

_HERE = Path(__file__).resolve().parent
DEFAULT_QUESTIONS = Path("./dataset/dataset_final/Questions_raw.json")
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_STEM = "gpt"


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


def _load_round1(path: Path) -> dict[str, dict[str, Any]]:
    """custom_id -> round-1 record (successful answers only), for --reflect."""
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    return {
        it["custom_id"]: it
        for it in items
        if it.get("custom_id") and it.get("answer") and not it.get("error")
    }


def _iter_jobs(items: list[QuestionItem], image_detail: str,
               round1: dict[str, dict[str, Any]] | None = None):
    """Yield (custom_id, item, schema, messages, round1_record | None).

    With `round1` set this is the self-reflection pass: items without a
    successful round-1 answer are skipped and the custom_id gains a `_r2`
    suffix so round 1 and round 2 can live in the same results file.
    """
    missing = 0
    for idx, item in enumerate(items):
        base_cid = _custom_id(item, idx)
        prev = None
        if round1 is not None:
            prev = round1.get(base_cid)
            if prev is None:
                missing += 1
                continue
        cid = f"{base_cid}_r2" if round1 is not None else base_cid
        frames = extract_frames(item.local_path)
        schema = schema_for(item.question)
        messages = build_messages(frames, item.question, schema,
                                  image_detail=image_detail, reflection=prev)
        yield cid, item, schema, messages, prev
    if missing:
        print(f"Skipped {missing} item(s) with no successful round-1 answer.")


def _reflection_fields(prev: dict[str, Any] | None, answer: str) -> dict[str, Any]:
    if prev is None:
        return {"history": []}
    return {
        "history": [prev],
        "evaluation_r1": prev.get("evaluation", ""),
        "answer_r1": prev.get("answer", ""),
        "changed": answer != prev.get("answer", ""),
    }


# ---------------------------------------------------------------------------
# Live mode
# ---------------------------------------------------------------------------

def run_live(
    items: list[QuestionItem],
    *,
    model: str,
    image_detail: str,
    output_path: Path,
    round1: dict[str, Any] | None = None,
) -> None:
    from openai import OpenAI

    client = OpenAI()
    results: list[dict[str, Any]] = []

    for n, (cid, item, schema, messages, prev) in enumerate(
            _iter_jobs(items, image_detail, round1)):
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
            "latency_s": round(dt, 3),
            "usage": {
                "input_tokens": usage.prompt_tokens,
                "cached_input_tokens": cached,
                "output_tokens": usage.completion_tokens,
            },
            "model": model,
            "image_detail": image_detail,
            "custom_id": cid,
            **_reflection_fields(prev, parsed.answer),
        }
        results.append(out)
        print(
            f"[{n + 1:>4}/{len(items)}] {dt:5.2f}s  "
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
    round1: dict[str, dict[str, Any]] | None = None,
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
        for cid, item, schema, messages, prev in _iter_jobs(items, image_detail, round1):
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
            if prev is not None:
                mapping[cid]["evaluation_r1"] = prev.get("evaluation", "")
                mapping[cid]["answer_r1"] = prev.get("answer", "")
                mapping[cid]["round1"] = prev
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
    round1: dict[str, dict[str, Any]] | None = None,
) -> None:
    from openai import OpenAI

    client = OpenAI()
    paths, mapping = _write_jsonl_chunks(
        items,
        image_detail=image_detail,
        model=model,
        jsonl_dir=jsonl_dir,
        max_bytes_per_file=max_bytes_per_file,
        round1=round1,
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
    ap.add_argument("--out-name", default=None,
                    help="Filename stem for the run's artifacts. Defaults to "
                         "`gpt` for the default model and to the sanitised "
                         "model id for any other, so two models never collide.")
    ap.add_argument("--reflect", action="store_true",
                    help="Round-2 self-reflection pass; requires --from.")
    ap.add_argument("--from", dest="from_results", type=Path, default=None,
                    help="[--reflect] Round-1 results JSON to re-examine.")
    ap.add_argument("--output", type=Path, default=None,
                    help="[live mode only] Where to save the results JSON "
                         "(default results/<out-name>_live_results.json).")
    ap.add_argument("--state", type=Path, default=None,
                    help="[batch mode only] Where to save the submission state "
                         "(default batches/batch_state.json for the default model).")
    ap.add_argument("--jsonl-dir", type=Path, default=None,
                    help="[batch mode only] Directory for batch input JSONL parts "
                         "(auto-chunked to stay under OpenAI's 200 MB file limit).")
    ap.add_argument("--max-bytes-per-file", type=int,
                    default=MAX_BYTES_PER_BATCH_FILE,
                    help="[batch mode only] Per-part JSONL size cap in bytes.")
    args = ap.parse_args()

    if args.reflect and args.from_results is None:
        sys.exit("--reflect needs --from <round-1 results json>.")
    if args.from_results is not None and not args.reflect:
        sys.exit("--from only makes sense together with --reflect.")

    stem = args.out_name or output_stem(
        args.model, default_model=DEFAULT_MODEL, default_stem=DEFAULT_STEM)
    suffix = "_r2" if args.reflect else ""
    if args.output is None:
        args.output = _HERE / "results" / f"{stem}_live_results{suffix}.json"
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
    print(f"Processing {len(items)} (video, question) pairs "
          f"with model={args.model}, detail={args.image_detail}"
          f"{', reflect' if args.reflect else ''}.")

    if args.mode == "live":
        run_live(items, model=args.model, image_detail=args.image_detail,
                 output_path=args.output, round1=round1)
    else:
        run_batch_submit(items, model=args.model, image_detail=args.image_detail,
                         state_path=args.state, jsonl_dir=args.jsonl_dir,
                         max_bytes_per_file=args.max_bytes_per_file,
                         round1=round1)


if __name__ == "__main__":
    main()
