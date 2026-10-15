"""Run the Claude evaluation — live, batch, or dry-run (nothing submitted).

Modes
-----
dry-run
    Build every request, write `batches/dry_run_requests.jsonl`, print a
    size / token estimate, submit nothing and contact no API. Use this first.
live
    Synchronous `client.messages.create` per request. Good for the pilot: you
    see the parse rate and `cache_read_input_tokens` within seconds. Results
    are flushed to disk after every item, so an interrupt loses nothing.
batch
    Message Batches API. Requests are sorted by video (so the ~4.6 questions of
    a clip sit next to each other and share the cached image prefix) and packed
    into batches of `--batch-size` requests / `--max-bytes-per-batch` bytes.
    Submits, records `batches/batch_state.json`, exits. Pick results up later
    with `retrieve_batch.py`.

Self-reflection variant
-----------------------
    python run_claude.py --mode batch --reflect --from results/claude_batch_results.json
Round 2 replays each round-1 request, appends the round-1 JSON as the assistant
turn and asks the model to re-examine. `custom_id` gets a `_r2` suffix; result
records gain `answer_r1`, `evaluation_r1`, `changed`, and `history=[round-1]`.

Output item shape (identical to the gpt/ and gemini/ backends):
    {
      "video":      <cluster-style path — the strip-prefix analysis code needs it>,
      "question":   <original question text>,
      "evaluation": <rationale string>,
      "answer":     <normalised final answer>,
      "history":    [] (or [round-1 record] when --reflect),
      "usage":      {"input_tokens", "cached_input_tokens",
                     "cache_creation_input_tokens", "output_tokens"},
      "model":      "claude-sonnet-5",
      "thinking":   "disabled" | "adaptive",
      "custom_id":  <stable id; also the batch key>,
    }
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv

from claude_request import (
    DEFAULT_MAX_TOKENS,
    batch_request,
    build_request_params,
    extract_usage,
    parse_result,
)
from gpt_helpers import QuestionItem, extract_frames, load_questions, output_stem, schema_for

_HERE = Path(__file__).resolve().parent
DEFAULT_QUESTIONS = Path("./dataset/dataset_final/Questions_raw.json")
DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_STEM = "claude"

# Anthropic's Message Batches API caps a batch at 100k requests / 256 MB. The
# byte cap is what bites here (5 base64 JPEGs ≈ 500 KB per request), so we chunk
# on both. 200 MB leaves headroom for the JSON envelope the SDK adds.
DEFAULT_BATCH_SIZE = 300
MAX_BYTES_PER_BATCH = 200 * 1024 * 1024

# Rough image-token estimate used by --mode dry-run only (Anthropic bills an
# image at about width * height / 750 tokens).
TOKENS_PER_PIXEL_DIVISOR = 750


# ---------------------------------------------------------------------------
# Environment / bookkeeping
# ---------------------------------------------------------------------------

def _load_env() -> None:
    for name in ("secrets.env", ".env"):
        p = _HERE / name
        if p.is_file():
            load_dotenv(p, override=False)
    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        sys.exit(
            f"ANTHROPIC_API_KEY missing - set it in {_HERE / '.env'} "
            f"or export it in your shell. See example.env."
        )


def _custom_id(item: QuestionItem, idx: int) -> str:
    # Same recipe as run_gpt.py / run_gemini.py, so a given (video, question)
    # pair carries the same id across all three backends.
    return f"q{idx:06d}_{Path(item.relative_path).stem[:40]}"


def _load_done_custom_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    return {
        it["custom_id"]
        for it in items
        if it.get("custom_id") and it.get("answer") and not it.get("error")
    }


def _raw_question_count(path: Path) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return len(json.load(f))


def _load_round1(path: Path) -> dict[str, dict[str, Any]]:
    """custom_id -> round-1 record, successful answers only."""
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    out: dict[str, dict[str, Any]] = {}
    for it in items:
        cid = it.get("custom_id")
        if cid and it.get("answer") and not it.get("error"):
            out[cid] = it
    return out


# ---------------------------------------------------------------------------
# Job construction
# ---------------------------------------------------------------------------

class Job:
    """One request, fully built and ready to send."""

    __slots__ = ("custom_id", "item", "schema", "params", "round1", "_nbytes")

    def __init__(self, custom_id: str, item: QuestionItem, schema, params: dict,
                 round1: dict | None):
        self.custom_id = custom_id
        self.item = item
        self.schema = schema
        self.params = params
        self.round1 = round1
        self._nbytes: int | None = None

    @property
    def nbytes(self) -> int:
        """Encoded size of this request as one batch entry (cached)."""
        if self._nbytes is None:
            self._nbytes = len(
                json.dumps(batch_request(self.custom_id, self.params),
                           ensure_ascii=False).encode("utf-8")
            )
        return self._nbytes


def _ordered_items(items: list[QuestionItem]) -> list[tuple[int, QuestionItem]]:
    """(original index, item) pairs, stably sorted so a video's rows are adjacent.

    The index is kept from the *unsorted* list: `custom_id` must stay identical
    to the GPT / Gemini runs on the same questions file.
    """
    return sorted(enumerate(items), key=lambda pair: pair[1].relative_path)


def iter_jobs(
    items: list[QuestionItem],
    *,
    model: str,
    thinking: str,
    effort: str | None,
    cache: bool,
    max_tokens: int,
    skip_cids: set[str],
    round1: dict[str, dict[str, Any]] | None,
) -> Iterator[Job]:
    skipped = 0
    no_round1 = 0
    for idx, item in _ordered_items(items):
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
        params = build_request_params(
            frames, item.question, schema, model,
            thinking=thinking, effort=effort, cache=cache,
            reflection=prev, max_tokens=max_tokens,
        )
        yield Job(cid, item, schema, params, prev)
    if skipped:
        print(f"Skipped {skipped} already-answered request(s) (--skip-existing).")
    if no_round1:
        print(f"Skipped {no_round1} item(s) with no successful round-1 answer.")


def _meta(job: Job, *, model: str, thinking: str, effort: str | None) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "video": job.item.cluster_path,
        "question": job.item.question,
        "schema": job.schema.__name__,
        "model": model,
        "thinking": thinking,
    }
    if effort is not None:
        meta["effort"] = effort
    if job.round1 is not None:
        meta["evaluation_r1"] = job.round1.get("evaluation", "")
        meta["answer_r1"] = job.round1.get("answer", "")
        meta["round1"] = job.round1
    return meta


def _result_item(job: Job, parsed: dict[str, str], usage: dict[str, int], *,
                 model: str, thinking: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "video": job.item.cluster_path,
        "question": job.item.question,
        "evaluation": parsed["evaluation"],
        "answer": parsed["answer"],
        "history": [],
        "usage": usage,
        "model": model,
        "thinking": thinking,
        "custom_id": job.custom_id,
    }
    if job.round1 is not None:
        out["evaluation_r1"] = job.round1.get("evaluation", "")
        out["answer_r1"] = job.round1.get("answer", "")
        out["changed"] = out["answer"] != out["answer_r1"]
        out["history"] = [job.round1]
    return out


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def _image_pixels(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        return im.size  # (w, h)


def run_dry(jobs: Iterator[Job], *, out_path: Path, model: str, thinking: str,
            batch_size: int, max_bytes: int, total_items: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    total_bytes = 0
    image_bytes = 0
    image_tokens = 0.0
    n_images = 0
    sample_dims: tuple[int, int] | None = None
    per_request_bytes: list[int] = []

    with open(out_path, "w", encoding="utf-8") as f:
        for job in jobs:
            line = json.dumps(batch_request(job.custom_id, job.params),
                              ensure_ascii=False) + "\n"
            nb = len(line.encode("utf-8"))
            f.write(line)
            n += 1
            total_bytes += nb
            per_request_bytes.append(nb)

            frames = extract_frames(job.item.local_path)
            for fr in frames:
                w, h = _image_pixels(fr.path)
                sample_dims = sample_dims or (w, h)
                image_bytes += fr.path.stat().st_size
                image_tokens += (w * h) / TOKENS_PER_PIXEL_DIVISOR
                n_images += 1

    if n == 0:
        print("Nothing to build — 0 requests.")
        return

    mean_bytes = total_bytes / n
    mean_img_bytes = image_bytes / n
    mean_img_tokens = image_tokens / n
    n_batches = max(1, -(-n // batch_size))

    print(f"\n=== dry run: {n} request(s) → {out_path} ===")
    print(f"model               : {model}  (thinking={thinking})")
    if sample_dims:
        print(f"frame size          : {sample_dims[0]}x{sample_dims[1]}  "
              f"({n_images / n:.1f} images/request)")
    print(f"JSONL total         : {total_bytes / 1e6:.2f} MB "
          f"(mean {mean_bytes / 1024:.1f} KB/request, "
          f"max {max(per_request_bytes) / 1024:.1f} KB)")
    print(f"raw JPEG bytes      : {image_bytes / 1e6:.2f} MB "
          f"(mean {mean_img_bytes / 1024:.1f} KB/request)")
    print(f"image tokens (w*h/{TOKENS_PER_PIXEL_DIVISOR}) : "
          f"{image_tokens:,.0f} total, {mean_img_tokens:,.0f}/request")
    print(f"batching            : {n_batches} batch(es) at --batch-size "
          f"{batch_size} / {max_bytes / 1e6:.0f} MB")
    if total_items and total_items != n:
        scale = total_items / n
        print(f"\nExtrapolated to the full {total_items} (video, question) pairs:")
        print(f"  JSONL          ~{total_bytes * scale / 1e9:.2f} GB")
        print(f"  batches        ~{max(1, -(-total_items // batch_size))} "
              f"at --batch-size {batch_size}")
        print(f"  image tokens   ~{image_tokens * scale / 1e6:.1f} M "
              "(before any cache discount)")
    print("\nNothing was submitted. Re-run with --mode batch (or --mode live) to send.")


# ---------------------------------------------------------------------------
# Live mode
# ---------------------------------------------------------------------------

def run_live(jobs: Iterator[Job], *, model: str, thinking: str, output_path: Path,
             sleep_s: float, max_retries: int) -> None:
    import anthropic

    client = anthropic.Anthropic(max_retries=max_retries)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Append-style: keep whatever is already in the file, keyed by custom_id.
    merged: dict[str, dict[str, Any]] = {}
    if output_path.is_file():
        with open(output_path, "r", encoding="utf-8") as f:
            for it in json.load(f):
                if it.get("custom_id"):
                    merged[it["custom_id"]] = it
        print(f"Loaded {len(merged)} prior result(s) from {output_path}.")

    def _flush() -> None:
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(list(merged.values()), fh, indent=2, ensure_ascii=False)

    n_ok = n_err = 0
    for i, job in enumerate(jobs):
        t0 = time.perf_counter()
        try:
            message = client.messages.create(**job.params)
            parsed = parse_result(message)
            usage = extract_usage(message)
            out = _result_item(job, parsed, usage, model=model, thinking=thinking)
            out["latency_s"] = round(time.perf_counter() - t0, 3)
            n_ok += 1
            print(
                f"[{i + 1:>4}] {out['latency_s']:5.2f}s  "
                f"in={usage['input_tokens']} "
                f"(cache_read={usage['cached_input_tokens']}, "
                f"cache_write={usage['cache_creation_input_tokens']})  "
                f"out={usage['output_tokens']}  -> {parsed['answer']}"
            )
        except Exception as exc:  # noqa: BLE001 — record and keep going
            n_err += 1
            out = {
                "video": job.item.cluster_path,
                "question": job.item.question,
                "evaluation": "",
                "answer": "",
                "history": [],
                "error": f"{type(exc).__name__}: {exc}",
                "model": model,
                "thinking": thinking,
                "custom_id": job.custom_id,
            }
            print(f"[{i + 1:>4}] ERROR {type(exc).__name__}: {exc}")
        merged[job.custom_id] = out
        _flush()
        if sleep_s:
            time.sleep(sleep_s)

    _flush()
    print(f"\n{n_ok} OK, {n_err} error(s); {len(merged)} total -> {output_path}")


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------

def chunk_jobs(jobs: list[Job], *, batch_size: int, max_bytes: int) -> list[list[Job]]:
    """Pack jobs into batches, never splitting a video across two batches.

    `jobs` must already be video-adjacent (see `_ordered_items`). Groups of the
    same video are kept whole so the cached image prefix is written once per
    clip inside a single batch.
    """
    if not jobs:
        return []

    groups: list[list[Job]] = []
    for job in jobs:
        if groups and groups[-1][0].item.relative_path == job.item.relative_path:
            groups[-1].append(job)
        else:
            groups.append([job])

    chunks: list[list[Job]] = []
    cur: list[Job] = []
    cur_bytes = 0
    for group in groups:
        g_bytes = sum(j.nbytes for j in group)
        too_many = cur and len(cur) + len(group) > batch_size
        too_big = cur and cur_bytes + g_bytes > max_bytes
        if too_many or too_big:
            chunks.append(cur)
            cur, cur_bytes = [], 0
        cur.extend(group)
        cur_bytes += g_bytes
        # A single video bigger than a whole batch: flush it on its own.
        while len(cur) > batch_size or cur_bytes > max_bytes:
            head, cur = cur[:batch_size], cur[batch_size:]
            chunks.append(head)
            cur_bytes = sum(j.nbytes for j in cur)
    if cur:
        chunks.append(cur)
    return chunks


def run_batch_submit(jobs: list[Job], *, model: str, thinking: str,
                     effort: str | None, cache: bool, state_path: Path,
                     batch_size: int, max_bytes: int) -> None:
    import anthropic

    if not jobs:
        print("Nothing new to submit.")
        return

    client = anthropic.Anthropic()
    chunks = chunk_jobs(jobs, batch_size=batch_size, max_bytes=max_bytes)
    print(f"{len(jobs)} request(s) → {len(chunks)} batch(es).")

    batches: list[dict[str, Any]] = []
    mapping: dict[str, dict[str, Any]] = {}
    for i, chunk in enumerate(chunks):
        payload = [batch_request(j.custom_id, j.params) for j in chunk]
        nbytes = sum(j.nbytes for j in chunk)
        batch = client.messages.batches.create(requests=payload)
        batches.append({
            "batch_id": batch.id,
            "n_requests": len(chunk),
            "bytes": nbytes,
            "status_at_submit": batch.processing_status,
            "created_at": str(getattr(batch, "created_at", "")),
        })
        for j in chunk:
            mapping[j.custom_id] = _meta(j, model=model, thinking=thinking, effort=effort)
        print(f"  [{i + 1}/{len(chunks)}] {len(chunk):>4} requests, "
              f"{nbytes / 1e6:6.1f} MB -> {batch.id}  ({batch.processing_status})")

    state: dict[str, Any] = {
        "model": model,
        "thinking": thinking,
        "effort": effort,
        "cache": cache,
        "batches": batches,
        "mapping": mapping,
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    # Append-style merge, like run_gemini.py: previous batches stay tracked so
    # retrieve_batch.py can still fetch them.
    if state_path.is_file():
        with open(state_path, "r", encoding="utf-8") as f:
            old = json.load(f)
        old_mapping = old.get("mapping", {})
        old_mapping.update(mapping)
        state["mapping"] = old_mapping
        state["batches"] = old.get("batches", []) + batches
        print(f"Merged with existing state at {state_path} "
              f"({len(state['batches'])} batches total).")
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

    print(f"\n{len(batches)} batch(es) submitted; state file: {state_path}")
    print(f"Run `python retrieve_batch.py --state {state_path} --wait` to fetch results.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=("live", "batch", "dry-run"), required=True)
    ap.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N questions (pilots / dry runs).")
    ap.add_argument("--model", default=os.environ.get("CLAUDE_MODEL", DEFAULT_MODEL))
    ap.add_argument("--thinking", choices=("disabled", "adaptive"), default="disabled",
                    help="`disabled` (default) matches the GPT/Gemini protocol.")
    ap.add_argument("--effort", choices=("low", "medium", "high", "xhigh", "max"),
                    default=None, help="output_config.effort; unset by default.")
    ap.add_argument("--no-cache", action="store_true",
                    help="Drop cache_control from the image prefix.")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--skip-existing", type=Path, default=None,
                    help="Results JSON whose successful custom_ids should be skipped.")
    ap.add_argument("--reflect", action="store_true",
                    help="Round-2 self-reflection pass; requires --from.")
    ap.add_argument("--from", dest="from_results", type=Path, default=None,
                    help="[--reflect] Round-1 results JSON to re-examine.")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                    help="[batch] Max requests per batch (default 300).")
    ap.add_argument("--max-bytes-per-batch", type=int, default=MAX_BYTES_PER_BATCH,
                    help="[batch] Max encoded bytes per batch (default 200 MB).")
    ap.add_argument("--sleep", type=float, default=0.2,
                    help="[live] Seconds to sleep between requests.")
    ap.add_argument("--max-retries", type=int, default=5,
                    help="[live] SDK retry budget per request.")
    ap.add_argument("--state", type=Path, default=None,
                    help="[batch] Submission state file "
                         "(default batches/batch_state.json, _r2 when --reflect).")
    ap.add_argument("--output", type=Path, default=None,
                    help="[live] Results JSON (default results/<stem>_live_results.json).")
    ap.add_argument("--dry-run-out", type=Path, default=None,
                    help="[dry-run] JSONL of built requests "
                         "(default batches/dry_run_requests.jsonl).")
    return ap


def resolve_paths(args: argparse.Namespace) -> None:
    """Fill in the model-aware defaults for --state / --output / --dry-run-out."""
    stem = output_stem(args.model, default_model=DEFAULT_MODEL, default_stem=DEFAULT_STEM)
    suffix = "_r2" if args.reflect else ""
    args.stem = stem
    if args.state is None:
        args.state = _HERE / "batches" / f"batch_state{suffix}.json"
    if args.output is None:
        args.output = _HERE / "results" / f"{stem}_live_results{suffix}.json"
    if args.dry_run_out is None:
        args.dry_run_out = _HERE / "batches" / f"dry_run_requests{suffix}.jsonl"


def main() -> None:
    args = build_parser().parse_args()
    if args.reflect and args.from_results is None:
        sys.exit("--reflect needs --from <round-1 results json>.")
    if args.from_results is not None and not args.reflect:
        sys.exit("--from only makes sense together with --reflect.")
    resolve_paths(args)

    if args.mode != "dry-run":
        _load_env()

    items = load_questions(args.questions)
    # Extrapolation target = every pair in the questions file, including the
    # ones whose clip is missing locally (they all resolve on the cluster).
    total_items = _raw_question_count(args.questions)
    if len(items) != total_items:
        print(f"{len(items)}/{total_items} pairs resolve to a local video.")
    if args.limit:
        items = items[: args.limit]

    round1 = _load_round1(args.from_results) if args.reflect else None
    if round1 is not None:
        print(f"Loaded {len(round1)} round-1 answer(s) from {args.from_results}.")
    skip_cids = _load_done_custom_ids(args.skip_existing) if args.skip_existing else set()
    if skip_cids:
        print(f"Loaded {len(skip_cids)} already-answered custom_ids from "
              f"{args.skip_existing}.")

    print(f"Processing {len(items)} (video, question) pairs with model={args.model}, "
          f"thinking={args.thinking}, effort={args.effort}, "
          f"cache={'off' if args.no_cache else 'on'}"
          f"{', reflect' if args.reflect else ''}.")

    jobs = iter_jobs(
        items,
        model=args.model,
        thinking=args.thinking,
        effort=args.effort,
        cache=not args.no_cache,
        max_tokens=args.max_tokens,
        skip_cids=skip_cids,
        round1=round1,
    )

    if args.mode == "dry-run":
        run_dry(jobs, out_path=args.dry_run_out, model=args.model,
                thinking=args.thinking, batch_size=args.batch_size,
                max_bytes=args.max_bytes_per_batch, total_items=total_items)
    elif args.mode == "live":
        run_live(jobs, model=args.model, thinking=args.thinking,
                 output_path=args.output, sleep_s=args.sleep,
                 max_retries=args.max_retries)
    else:
        run_batch_submit(list(jobs), model=args.model, thinking=args.thinking,
                         effort=args.effort, cache=not args.no_cache,
                         state_path=args.state, batch_size=args.batch_size,
                         max_bytes=args.max_bytes_per_batch)


if __name__ == "__main__":
    main()
