"""Build Anthropic Messages API request params for the VLM-eval dataset.

One request = one (video, question) pair. Layout (ordered):

    system                          : constant across *every* request  ─┐
    messages[0].content[0..N-1]     : the N frames of the video         │ cached
                                      (last one carries cache_control)  ┘ prefix
    messages[0].content[N]          : question text + allowed answers   — variable

Anthropic's prompt cache is a prefix match, so putting the images first and
marking the *last* image block with `cache_control` makes the 5 frames of a clip
the shared cached prefix for all ~4.6 questions of that clip. Only the trailing
question text differs, which is exactly what we want billed at full price.

Structured output uses `output_config.format` (`type: "json_schema"`), which
guarantees the first text block of the response is valid JSON matching the
schema. `additionalProperties: False` is required and pydantic does not emit it,
so we add it ourselves — same fix as the GPT and Gemini backends.

The result of `build_request_params` is a plain dict, usable both as
`client.messages.create(**params)` and as the `params` of a batch `Request`.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Literal

from gpt_helpers import (
    REFLECTION_PROMPT,
    SYSTEM_PROMPT,
    Frame,
    _Base,
    build_question_text,
    round1_assistant_text,
)

ThinkingMode = Literal["disabled", "adaptive"]
Effort = Literal["low", "medium", "high", "xhigh", "max"]

# The schema is two short fields; 512 output tokens is plenty even with a
# 1-3 sentence rationale. Adaptive thinking needs headroom, so callers that
# switch `--thinking adaptive` should raise this.
DEFAULT_MAX_TOKENS = 512

MEDIA_TYPE = "image/jpeg"


def _image_block(frame: Frame, *, cache: bool = False) -> dict[str, Any]:
    """One base64 image content block, optionally the cache breakpoint."""
    raw = frame.path.read_bytes()
    block: dict[str, Any] = {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": MEDIA_TYPE,
            "data": base64.b64encode(raw).decode("ascii"),
        },
    }
    if cache:
        block["cache_control"] = {"type": "ephemeral"}
    return block


def build_output_format(schema_cls: type[_Base]) -> dict[str, Any]:
    schema_json = schema_cls.model_json_schema()
    schema_json["additionalProperties"] = False
    return {"type": "json_schema", "schema": schema_json}


def build_request_params(
    frames: list[Frame],
    question: str,
    schema_cls: type[_Base],
    model: str,
    *,
    thinking: ThinkingMode = "disabled",
    effort: Effort | None = None,
    cache: bool = True,
    reflection: dict[str, Any] | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict[str, Any]:
    """Return the kwargs for one `client.messages.create(...)` call.

    thinking
        "disabled" (default) mirrors the GPT / Gemini protocol — no extended
        thinking, so the three closed-source arms are directly comparable.
        "adaptive" is the optional extra arm; pair it with `effort`.
    effort
        Maps to `output_config.effort` (low | medium | high | xhigh | max).
        Left unset when None so the model's own default applies.
    cache
        Put `cache_control: {"type": "ephemeral"}` on the last image block, so
        system + frames form the cached prefix shared by every question of the
        same clip. `--no-cache` turns this off (useful for a clean cost A/B).
    reflection
        Round-1 result record (`evaluation`, `answer`). When given, the message
        list becomes user(images+question) / assistant(round-1 JSON) /
        user(re-examine) — the round-2 self-reflection variant.
    """
    if not frames:
        raise ValueError("build_request_params needs at least one frame")

    last = len(frames) - 1
    content: list[dict[str, Any]] = [
        _image_block(f, cache=cache and i == last) for i, f in enumerate(frames)
    ]
    content.append({"type": "text", "text": build_question_text(question, schema_cls)})

    messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
    if reflection is not None:
        messages.append({
            "role": "assistant",
            "content": round1_assistant_text(
                reflection.get("evaluation", ""), reflection.get("answer", "")
            ),
        })
        messages.append({"role": "user", "content": REFLECTION_PROMPT})

    output_config: dict[str, Any] = {"format": build_output_format(schema_cls)}
    if effort is not None:
        output_config["effort"] = effort

    params: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": SYSTEM_PROMPT,
        "messages": messages,
        "output_config": output_config,
    }
    if thinking == "disabled":
        params["thinking"] = {"type": "disabled"}
    elif thinking == "adaptive":
        # `display: "omitted"` is the default on Sonnet 5 / Opus 5; we never
        # read the thinking text, so leave it alone and keep payloads small.
        params["thinking"] = {"type": "adaptive"}
    else:
        raise ValueError(f"Unknown thinking mode: {thinking!r}")
    return params


def batch_request(custom_id: str, params: dict[str, Any]) -> dict[str, Any]:
    """One entry of `client.messages.batches.create(requests=[...])`.

    `Request` / `MessageCreateParamsNonStreaming` are TypedDicts, i.e. plain
    dicts at runtime, so the returned value is both a valid SDK argument and
    JSON-serialisable (which is what `--mode dry-run` writes out).
    """
    return {"custom_id": custom_id, "params": params}


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _content_blocks(message: Any) -> list[Any]:
    if isinstance(message, dict):
        return message.get("content") or []
    return getattr(message, "content", None) or []


def _block_field(block: Any, name: str) -> Any:
    if isinstance(block, dict):
        return block.get(name)
    return getattr(block, name, None)


def first_text(message: Any) -> str:
    """The first `text` content block of a Message (thinking blocks skipped)."""
    for block in _content_blocks(message):
        if _block_field(block, "type") == "text":
            return _block_field(block, "text") or ""
    raise ValueError("Message has no text content block")


def parse_result(message: Any) -> dict[str, str]:
    """Structured answer of one Message → {"evaluation": ..., "answer": ...}.

    `output_config.format` guarantees the first text block is valid JSON, so a
    plain `json.loads` is enough; we only tolerate surrounding whitespace.
    Missing keys are a hard error — silently emitting an empty answer would
    quietly poison the accuracy numbers downstream.
    """
    text = first_text(message).strip()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Structured output was not a JSON object: {text[:200]!r}")
    missing = [k for k in ("evaluation", "answer") if k not in data]
    if missing:
        raise KeyError(f"Structured output missing key(s) {missing}: {text[:200]!r}")
    return {"evaluation": data["evaluation"], "answer": data["answer"]}


def extract_usage(message: Any) -> dict[str, int]:
    """Usage in the shape the gpt/gemini result JSONs already use."""
    usage = message.get("usage") if isinstance(message, dict) else getattr(message, "usage", None)
    if usage is None:
        return {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "output_tokens": 0,
        }

    def _get(name: str) -> int:
        val = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
        return int(val or 0)

    return {
        "input_tokens": _get("input_tokens"),
        # `cached_input_tokens` is the cross-backend name for "served from cache".
        "cached_input_tokens": _get("cache_read_input_tokens"),
        "cache_creation_input_tokens": _get("cache_creation_input_tokens"),
        "output_tokens": _get("output_tokens"),
    }
