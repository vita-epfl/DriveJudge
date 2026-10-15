"""Offline checks for the Claude backend — plain asserts, no network.

Run from this directory with the repo venv:

    cd src/evaluation/closed_source/claude
    ../../../../.venv/bin/python test_offline.py

Covers:
  1. request-param shape (image count, cache_control placement, schema strictness,
     thinking disabled by default, adaptive + effort arm)
  2. parse_result / extract_usage on fake message objects
  3. batch chunking (sizes, video adjacency, byte cap)
  4. reflection message construction for Claude, GPT and Gemini
  5. `--mode dry-run --limit 20` end-to-end on the real frame cache, printing the
     per-request image bytes and the w*h/750 token estimate
  6. live mode against a monkeypatched `anthropic.Anthropic`, to exercise the
     incremental write path

Nothing here contacts an API: (5) only builds requests and (6) mocks the client.
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import claude_request  # noqa: E402
import run_claude  # noqa: E402
from claude_request import (  # noqa: E402
    batch_request,
    build_request_params,
    extract_usage,
    parse_result,
)
from gpt_helpers import Frame, extract_frames, load_questions, schema_for  # noqa: E402

REPO_ROOT = _HERE.parents[3]
QUESTIONS = REPO_ROOT / "dataset/dataset_final/Questions_raw.json"

_PASSED: list[str] = []


def ok(name: str) -> None:
    _PASSED.append(name)
    print(f"  ok  {name}")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeBlock:
    def __init__(self, type_: str, text: str = "", thinking: str = ""):
        self.type = type_
        self.text = text
        self.thinking = thinking


class FakeUsage:
    def __init__(self, **kw):
        self.input_tokens = kw.get("input_tokens", 0)
        self.output_tokens = kw.get("output_tokens", 0)
        self.cache_read_input_tokens = kw.get("cache_read_input_tokens", 0)
        self.cache_creation_input_tokens = kw.get("cache_creation_input_tokens", 0)


class FakeMessage:
    def __init__(self, blocks, usage=None, stop_reason="end_turn"):
        self.content = blocks
        self.usage = usage or FakeUsage()
        self.stop_reason = stop_reason


def _fake_frames(tmp: Path, n: int = 5) -> list[Frame]:
    frames = []
    for i in range(n):
        p = tmp / f"frame_{i:02d}.jpg"
        p.write_bytes(bytes([i]) * (100 + i))
        frames.append(Frame(i, p, "data:image/jpeg;base64,"))
    return frames


# ---------------------------------------------------------------------------
# 1. Request params
# ---------------------------------------------------------------------------

def test_request_params(tmp: Path) -> None:
    print("\n[1] request params")
    from schemas import YesNoAnswer

    frames = _fake_frames(tmp, 5)
    params = build_request_params(
        frames, "Is the ego car on a highway ?", YesNoAnswer, "claude-sonnet-5")

    content = params["messages"][0]["content"]
    images = [b for b in content if b["type"] == "image"]
    texts = [b for b in content if b["type"] == "text"]
    assert len(images) == 5, f"expected 5 image blocks, got {len(images)}"
    assert len(texts) == 1 and texts[-1] is content[-1], "question text must come last"
    ok("5 image blocks, question text last")

    assert all(b["source"]["type"] == "base64" for b in images)
    assert all(b["source"]["media_type"] == "image/jpeg" for b in images)
    ok("image blocks are base64/image/jpeg")

    cached = [i for i, b in enumerate(content) if "cache_control" in b]
    assert cached == [4], f"cache_control must sit on the LAST image only, got {cached}"
    assert content[4]["cache_control"] == {"type": "ephemeral"}
    ok("cache_control on the last image block only")

    schema = params["output_config"]["format"]["schema"]
    assert params["output_config"]["format"]["type"] == "json_schema"
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == {"evaluation", "answer"}
    ok("output_config.format is a strict json_schema")

    assert params["thinking"] == {"type": "disabled"}, params.get("thinking")
    assert "effort" not in params["output_config"]
    ok("thinking disabled by default, no effort")

    assert params["system"] == claude_request.SYSTEM_PROMPT
    assert "expert video analyst" in params["system"]
    assert params["max_tokens"] == claude_request.DEFAULT_MAX_TOKENS == 512
    ok("system prompt reused verbatim from gpt/prompt_builder, max_tokens=512")

    q = texts[0]["text"]
    assert q.startswith("Question: Is the ego car on a highway ?")
    assert "Allowed answers: ['Yes', 'No']" in q
    assert q.endswith("Respond using the structured schema.")
    ok("question text matches the gemini/gpt wording")

    # Optional arm: adaptive thinking + effort.
    adaptive = build_request_params(
        frames, "Is the ego car on a highway ?", YesNoAnswer, "claude-sonnet-5",
        thinking="adaptive", effort="high")
    assert adaptive["thinking"] == {"type": "adaptive"}
    assert adaptive["output_config"]["effort"] == "high"
    ok("--thinking adaptive --effort high arm")

    # --no-cache
    nocache = build_request_params(
        frames, "Is the ego car on a highway ?", YesNoAnswer, "claude-sonnet-5",
        cache=False)
    assert not any("cache_control" in b for b in nocache["messages"][0]["content"])
    ok("cache=False drops every cache_control")

    # batch_request is JSON-serialisable and keyed by custom_id.
    req = batch_request("q000001_abc", params)
    assert req["custom_id"] == "q000001_abc" and req["params"] is params
    json.dumps(req)
    ok("batch_request is JSON-serialisable")


# ---------------------------------------------------------------------------
# 2. Response parsing
# ---------------------------------------------------------------------------

def test_parse_result() -> None:
    print("\n[2] parse_result / extract_usage")
    payload = '{"evaluation": "Lane markings are consistent.", "answer": "Yes"}'
    msg = FakeMessage([FakeBlock("text", payload + "\n  ")])
    parsed = parse_result(msg)
    assert parsed == {"evaluation": "Lane markings are consistent.", "answer": "Yes"}
    ok("parses the first text block, tolerates trailing whitespace")

    with_thinking = FakeMessage([FakeBlock("thinking", thinking="hmm"),
                                 FakeBlock("text", payload)])
    assert parse_result(with_thinking)["answer"] == "Yes"
    ok("skips a leading thinking block")

    assert parse_result({"content": [{"type": "text", "text": payload}]})["answer"] == "Yes"
    ok("accepts a dict-shaped message too")

    for bad, exc in (
        ('{"evaluation": "x"}', KeyError),
        ('{"answer": "Yes"}', KeyError),
        ('["not", "an", "object"]', ValueError),
        ("not json at all", json.JSONDecodeError),
    ):
        try:
            parse_result(FakeMessage([FakeBlock("text", bad)]))
        except exc:
            pass
        else:
            raise AssertionError(f"expected {exc.__name__} for {bad!r}")
    ok("raises on missing keys / non-object / non-JSON")

    try:
        parse_result(FakeMessage([FakeBlock("thinking", thinking="only")]))
    except ValueError:
        ok("raises when there is no text block")
    else:
        raise AssertionError("expected ValueError for a message with no text block")

    usage = extract_usage(FakeMessage(
        [FakeBlock("text", payload)],
        FakeUsage(input_tokens=120, output_tokens=44,
                  cache_read_input_tokens=5300, cache_creation_input_tokens=0)))
    assert usage == {"input_tokens": 120, "cached_input_tokens": 5300,
                     "cache_creation_input_tokens": 0, "output_tokens": 44}
    ok("usage maps cache_read_input_tokens -> cached_input_tokens")


# ---------------------------------------------------------------------------
# 3. Batch chunking
# ---------------------------------------------------------------------------

class StubItem:
    def __init__(self, video: str):
        self.relative_path = video
        self.cluster_path = "/mnt/.../final_dataset/" + video
        self.question = "q?"
        self.local_path = Path("/dev/null")


def _stub_jobs(spec: list[tuple[str, int]], nbytes: int = 1000) -> list[run_claude.Job]:
    jobs = []
    for video, count in spec:
        for _ in range(count):
            j = run_claude.Job("cid", StubItem(video), None, {}, None)
            j._nbytes = nbytes
            jobs.append(j)
    return jobs


def test_chunking() -> None:
    print("\n[3] batch chunking")
    big = 10 ** 9

    jobs = _stub_jobs([(f"v{i:03d}.mp4", 5) for i in range(200)])  # 1000 requests
    chunks = run_claude.chunk_jobs(jobs, batch_size=300, max_bytes=big)
    assert sum(len(c) for c in chunks) == 1000
    assert all(len(c) <= 300 for c in chunks), [len(c) for c in chunks]
    assert len(chunks) == 4, [len(c) for c in chunks]
    ok(f"1000 requests -> {[len(c) for c in chunks]} at batch-size 300")

    for c in chunks:
        videos = [j.item.relative_path for j in c]
        # A video must never be split across chunks: each video appears in
        # exactly one chunk, and contiguously inside it.
        seen, runs = set(), []
        for v in videos:
            if not runs or runs[-1] != v:
                runs.append(v)
                assert v not in seen, f"{v} is not contiguous inside its chunk"
                seen.add(v)
    all_chunk_of = {}
    for i, c in enumerate(chunks):
        for j in c:
            prev = all_chunk_of.setdefault(j.item.relative_path, i)
            assert prev == i, f"{j.item.relative_path} split across chunks {prev}/{i}"
    ok("every video stays whole and contiguous inside one batch")

    # Byte cap wins over the count cap.
    jobs = _stub_jobs([(f"v{i:03d}.mp4", 5) for i in range(20)], nbytes=10_000_000)
    chunks = run_claude.chunk_jobs(jobs, batch_size=300, max_bytes=100_000_000)
    assert sum(len(c) for c in chunks) == 100
    assert all(sum(j.nbytes for j in c) <= 100_000_000 for c in chunks)
    assert len(chunks) == 10, [len(c) for c in chunks]
    ok("byte cap splits before the count cap")

    # A single video larger than a whole batch is flushed on its own.
    jobs = _stub_jobs([("huge.mp4", 7)])
    chunks = run_claude.chunk_jobs(jobs, batch_size=3, max_bytes=big)
    assert [len(c) for c in chunks] == [3, 3, 1]
    ok("an over-sized video is split rather than dropped")

    assert run_claude.chunk_jobs([], batch_size=300, max_bytes=big) == []
    ok("empty input -> no chunks")

    # `_ordered_items` groups by video while keeping the original index.
    items = [StubItem("b.mp4"), StubItem("a.mp4"), StubItem("b.mp4"), StubItem("a.mp4")]
    ordered = run_claude._ordered_items(items)
    assert [i for i, _ in ordered] == [1, 3, 0, 2]
    assert [it.relative_path for _, it in ordered] == ["a.mp4", "a.mp4", "b.mp4", "b.mp4"]
    ok("_ordered_items sorts by video and keeps the original index for custom_id")


# ---------------------------------------------------------------------------
# 4. Reflection prompts, all three backends
# ---------------------------------------------------------------------------

def test_reflection(tmp: Path) -> None:
    print("\n[4] reflection message construction")
    from prompt_builder import REFLECTION_PROMPT, build_messages
    from schemas import YesNoAnswer

    sys.path.insert(0, str(_HERE.parent / "gemini"))
    import gemini_request  # noqa: E402

    frames = _fake_frames(tmp, 5)
    r1 = {"evaluation": "The road looks like a highway.", "answer": "Yes",
          "custom_id": "q000001_x"}
    expected_assistant = json.dumps(
        {"evaluation": r1["evaluation"], "answer": r1["answer"]}, ensure_ascii=False)

    # --- Claude ---
    base = build_request_params(frames, "Is the ego car on a highway ?",
                                YesNoAnswer, "claude-sonnet-5")
    ref = build_request_params(frames, "Is the ego car on a highway ?",
                               YesNoAnswer, "claude-sonnet-5", reflection=r1)
    msgs = ref["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[0] == base["messages"][0], "round-1 prefix must be byte-identical"
    assert msgs[1]["content"] == expected_assistant
    assert msgs[2]["content"] == REFLECTION_PROMPT
    assert ref["output_config"] == base["output_config"], "same schema in round 2"
    ok("claude: user(images+q) / assistant(round-1 JSON) / user(re-examine)")

    # --- GPT ---
    gbase = build_messages(frames, "Is the ego car on a highway ?", YesNoAnswer)
    gref = build_messages(frames, "Is the ego car on a highway ?", YesNoAnswer,
                          reflection=r1)
    assert [m["role"] for m in gbase] == ["system", "user"], "round 1 unchanged"
    assert [m["role"] for m in gref] == ["system", "user", "assistant", "user"]
    assert gref[:2] == gbase
    assert gref[2]["content"] == expected_assistant
    assert gref[3]["content"] == REFLECTION_PROMPT
    ok("gpt: system / user / assistant(round-1 JSON) / user(re-examine)")

    # --- Gemini ---
    mbase = gemini_request.build_request_body(
        frames, "Is the ego car on a highway ?", YesNoAnswer)
    mref = gemini_request.build_request_body(
        frames, "Is the ego car on a highway ?", YesNoAnswer, reflection=r1)
    assert [c["role"] for c in mbase["contents"]] == ["user"], "round 1 unchanged"
    assert [c["role"] for c in mref["contents"]] == ["user", "model", "user"]
    assert mref["contents"][0] == mbase["contents"][0]
    assert mref["contents"][1]["parts"][0]["text"] == expected_assistant
    assert mref["contents"][2]["parts"][0]["text"] == REFLECTION_PROMPT
    assert mref["generation_config"] == mbase["generation_config"]
    ok("gemini: user / model(round-1 JSON) / user(re-examine)")

    # custom_id stability: `<round-1 id>_r2`.
    r1_file = tmp / "round1.json"
    r1_file.write_text(json.dumps([
        {"custom_id": "q000000_a", "answer": "Yes", "evaluation": "e"},
        {"custom_id": "q000001_b", "answer": "", "evaluation": "e"},   # unanswered
        {"custom_id": "q000002_c", "answer": "No", "evaluation": "e", "error": "boom"},
    ]))
    loaded = run_claude._load_round1(r1_file)
    assert set(loaded) == {"q000000_a"}, loaded
    ok("_load_round1 keeps only successful answers")

    job = run_claude.Job("q000000_a_r2", StubItem("a.mp4"), YesNoAnswer, {},
                         loaded["q000000_a"])
    rec = run_claude._result_item(job, {"evaluation": "e2", "answer": "No"},
                                  {"input_tokens": 1}, model="claude-sonnet-5",
                                  thinking="disabled")
    assert rec["custom_id"] == "q000000_a_r2"
    assert rec["answer_r1"] == "Yes" and rec["evaluation_r1"] == "e"
    assert rec["changed"] is True
    assert rec["history"] == [loaded["q000000_a"]]
    unchanged = run_claude._result_item(job, {"evaluation": "e2", "answer": "Yes"},
                                        {}, model="m", thinking="disabled")
    assert unchanged["changed"] is False
    ok("round-2 records carry answer_r1 / evaluation_r1 / changed / history")

    # GPT's equivalent helper.
    import run_gpt  # noqa: E402
    assert run_gpt._reflection_fields(None, "Yes") == {"history": []}
    f = run_gpt._reflection_fields(r1, "No")
    assert f["answer_r1"] == "Yes" and f["changed"] is True and f["history"] == [r1]
    ok("gpt _reflection_fields mirrors the claude record shape")


# ---------------------------------------------------------------------------
# 5. dry-run end-to-end on the real frame cache
# ---------------------------------------------------------------------------

def test_dry_run(tmp: Path) -> None:
    print("\n[5] --mode dry-run --limit 20 (real frames, nothing submitted)")
    if not QUESTIONS.is_file():
        print(f"  SKIP  {QUESTIONS} not found")
        return
    items = load_questions(QUESTIONS)
    if len(items) < 20:
        print(f"  SKIP  only {len(items)} items resolve to a local video")
        return

    out = tmp / "dry_run_requests.jsonl"
    argv = sys.argv
    sys.argv = ["run_claude.py", "--mode", "dry-run", "--limit", "20",
                "--questions", str(QUESTIONS), "--dry-run-out", str(out)]
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            run_claude.main()
    finally:
        sys.argv = argv
    report = buf.getvalue()
    print("\n".join("      " + ln for ln in report.strip().splitlines()))

    lines = out.read_text().strip().splitlines()
    assert len(lines) == 20, f"expected 20 JSONL lines, got {len(lines)}"
    ok("wrote 20 request lines and submitted nothing")

    first = json.loads(lines[0])
    assert set(first) == {"custom_id", "params"}
    content = first["params"]["messages"][0]["content"]
    assert sum(1 for b in content if b["type"] == "image") == 5
    assert "cache_control" in content[4] and content[-1]["type"] == "text"
    assert first["params"]["thinking"] == {"type": "disabled"}
    assert first["params"]["output_config"]["format"]["schema"]["additionalProperties"] is False
    ok("each line is a well-formed batch request built from the cached JPEGs")

    cids = [json.loads(ln)["custom_id"] for ln in lines]
    assert len(set(cids)) == 20, "custom_ids must be unique"
    videos = [json.loads(ln)["params"]["messages"][0]["content"][-1]["text"] for ln in lines]
    assert len(videos) == 20
    ok("custom_ids unique")

    # Frames really came from gpt/.frame_cache (extracted on demand if missing).
    cache_dir = (_HERE.parent / "gpt" / ".frame_cache").resolve()
    frames = extract_frames(items[0].local_path)
    assert all(str(f.path).startswith(str(cache_dir)) for f in frames), \
        f"frames must live under {cache_dir}"
    assert all(f.path.is_file() for f in frames)
    ok(f"frames served from {cache_dir}")

    # Per-request image bytes + token estimate, printed for the report.
    from PIL import Image
    total_img_bytes = 0
    total_tokens = 0.0
    for i in range(20):
        it = items[i]
        for fr in extract_frames(it.local_path):
            total_img_bytes += fr.path.stat().st_size
            with Image.open(fr.path) as im:
                w, h = im.size
            total_tokens += (w * h) / run_claude.TOKENS_PER_PIXEL_DIVISOR
    jsonl_bytes = out.stat().st_size
    print(f"\n      --- 20-request estimate ---")
    print(f"      image bytes / request : {total_img_bytes / 20 / 1024:8.1f} KB")
    print(f"      JSONL bytes / request : {jsonl_bytes / 20 / 1024:8.1f} KB")
    print(f"      image tokens / request: {total_tokens / 20:8.0f} "
          f"(w*h/{run_claude.TOKENS_PER_PIXEL_DIVISOR}, 5 frames)")
    n_all = len(json.loads(QUESTIONS.read_text()))
    print(f"      extrapolated to {n_all}: "
          f"{jsonl_bytes / 20 * n_all / 1e9:.2f} GB JSONL, "
          f"{total_tokens / 20 * n_all / 1e6:.1f} M image tokens")
    ok("printed the per-request size / token estimate")


# ---------------------------------------------------------------------------
# 6. Live mode against a mocked client
# ---------------------------------------------------------------------------

def test_live_smoke(tmp: Path) -> None:
    print("\n[6] live mode write path (mocked anthropic.Anthropic)")
    import anthropic
    from schemas import YesNoAnswer

    calls: list[dict] = []

    class FakeMessages:
        def create(self, **params):
            calls.append(params)
            payload = json.dumps({"evaluation": "looks like a highway",
                                  "answer": "Yes"})
            return FakeMessage(
                [FakeBlock("text", payload)],
                FakeUsage(input_tokens=90, output_tokens=30,
                          cache_read_input_tokens=5100 if len(calls) > 1 else 0,
                          cache_creation_input_tokens=0 if len(calls) > 1 else 5100),
            )

    class FakeClient:
        def __init__(self, **kw):
            self.messages = FakeMessages()

    real = anthropic.Anthropic
    anthropic.Anthropic = FakeClient  # type: ignore[assignment]
    try:
        frames = _fake_frames(tmp, 5)
        items = [StubItem("v1.mp4"), StubItem("v1.mp4")]
        jobs = [
            run_claude.Job(
                f"q00000{i}_v1", it, YesNoAnswer,
                build_request_params(frames, "Is the ego car on a highway ?",
                                     YesNoAnswer, "claude-sonnet-5"),
                None)
            for i, it in enumerate(items)
        ]
        out_path = tmp / "results" / "claude_live_results.json"
        buf = io.StringIO()
        with redirect_stdout(buf):
            run_claude.run_live(iter(jobs), model="claude-sonnet-5",
                                thinking="disabled", output_path=out_path,
                                sleep_s=0, max_retries=1)
    finally:
        anthropic.Anthropic = real  # type: ignore[assignment]

    assert len(calls) == 2, calls
    assert calls[0]["model"] == "claude-sonnet-5"
    ok("client.messages.create called once per job with the built params")

    saved = json.loads(out_path.read_text())
    assert len(saved) == 2
    rec = saved[0]
    assert set(rec) >= {"video", "question", "evaluation", "answer", "history",
                        "usage", "model", "thinking", "custom_id"}
    assert rec["answer"] == "Yes" and rec["history"] == []
    assert rec["thinking"] == "disabled"
    assert set(rec["usage"]) == {"input_tokens", "cached_input_tokens",
                                 "cache_creation_input_tokens", "output_tokens"}
    assert saved[1]["usage"]["cached_input_tokens"] == 5100, \
        "2nd request of a video should read from cache"
    ok("results JSON has the shared gpt/gemini record shape")

    # Re-running merges instead of truncating.
    anthropic.Anthropic = FakeClient  # type: ignore[assignment]
    try:
        with redirect_stdout(io.StringIO()):
            run_claude.run_live(iter(jobs[:1]), model="claude-sonnet-5",
                                thinking="disabled", output_path=out_path,
                                sleep_s=0, max_retries=1)
    finally:
        anthropic.Anthropic = real  # type: ignore[assignment]
    assert len(json.loads(out_path.read_text())) == 2, "append-merge by custom_id"
    ok("re-running merges by custom_id instead of truncating")

    # An API failure is recorded, not fatal.
    class BoomMessages:
        def create(self, **params):
            raise RuntimeError("503 overloaded")

    class BoomClient:
        def __init__(self, **kw):
            self.messages = BoomMessages()

    anthropic.Anthropic = BoomClient  # type: ignore[assignment]
    try:
        boom_path = tmp / "results" / "boom.json"
        with redirect_stdout(io.StringIO()):
            run_claude.run_live(iter(jobs[:1]), model="claude-sonnet-5",
                                thinking="disabled", output_path=boom_path,
                                sleep_s=0, max_retries=1)
    finally:
        anthropic.Anthropic = real  # type: ignore[assignment]
    err = json.loads(boom_path.read_text())[0]
    assert err["answer"] == "" and "503 overloaded" in err["error"]
    ok("a failing request is recorded with an `error` field and does not abort")


# ---------------------------------------------------------------------------
# 7. Output naming
# ---------------------------------------------------------------------------

def test_naming() -> None:
    print("\n[7] model-aware output names")
    from naming import output_stem, sanitise_model

    assert output_stem("gpt-5.4-mini", default_model="gpt-5.4-mini",
                       default_stem="gpt") == "gpt"
    assert output_stem("gpt-5.5", default_model="gpt-5.4-mini",
                       default_stem="gpt") == "gpt-5.5"
    assert output_stem("gemini-3-flash-preview",
                       default_model="gemini-3-flash-preview",
                       default_stem="gemini") == "gemini"
    assert output_stem("claude-sonnet-5", default_model="claude-sonnet-5",
                       default_stem="claude") == "claude"
    assert output_stem("claude-opus-5", default_model="claude-sonnet-5",
                       default_stem="claude") == "claude-opus-5"
    assert sanitise_model("us.anthropic/claude opus:5") == "us.anthropic_claude_opus_5"
    ok("default model keeps the historical stem, others carry the model id")

    ns = run_claude.build_parser().parse_args(["--mode", "dry-run"])
    run_claude.resolve_paths(ns)
    assert ns.output.name == "claude_live_results.json"
    assert ns.state.name == "batch_state.json"
    assert ns.dry_run_out.name == "dry_run_requests.jsonl"

    ns = run_claude.build_parser().parse_args(
        ["--mode", "batch", "--model", "claude-opus-5"])
    run_claude.resolve_paths(ns)
    assert ns.output.name == "claude-opus-5_live_results.json"

    ns = run_claude.build_parser().parse_args(
        ["--mode", "batch", "--reflect", "--from", "x.json"])
    run_claude.resolve_paths(ns)
    assert ns.state.name == "batch_state_r2.json"
    assert ns.output.name == "claude_live_results_r2.json"
    ok("run_claude default paths: model- and round-aware")

    sys.path.insert(0, str(_HERE / "analysis"))
    import run_analysis as claude_analysis  # noqa: E402
    assert claude_analysis.display_name("claude-sonnet-5") == "Claude-sonnet-5"
    assert claude_analysis.display_name("claude-opus-5") == "Claude-opus-5"
    a = claude_analysis.parse_args([])
    assert a.analyzed_out.name == "Claude-sonnet-5_analyzed.parquet"
    assert a.analyzed_out.parent.name == "results"
    assert a.results.name == "claude_batch_results.json"
    assert a.out_name == "claude"
    ok("analysis writes results/Claude-<model>_analyzed.parquet")


# ---------------------------------------------------------------------------

def main() -> int:
    os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_request_params(tmp)
        test_parse_result()
        test_chunking()
        test_reflection(tmp)
        test_dry_run(tmp)
        test_live_smoke(tmp)
        test_naming()
    print(f"\n{len(_PASSED)} checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
