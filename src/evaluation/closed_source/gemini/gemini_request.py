"""Build Gemini Batch API request bodies (JSONL line shape).

Each line of the JSONL file uploaded to Gemini's File API has shape:

    {
      "key": "<custom id>",
      "request": {
        "system_instruction": {"parts": [{"text": "..."}]},
        "contents": [
          {
            "role": "user",
            "parts": [
              {"inline_data": {"mime_type": "image/jpeg", "data": "<b64>"}},
              ...,
              {"text": "Question: ...\\nAllowed answers: ...\\nRespond using the structured schema."}
            ]
          }
        ],
        "generation_config": {
          "response_mime_type": "application/json",
          "response_json_schema": {... pydantic JSON schema ...}
        }
      }
    }
"""

from __future__ import annotations

import base64
from typing import Any

from gpt_helpers import (
    REFLECTION_PROMPT,
    Frame,
    _Base,
    allowed_answers,
    round1_assistant_text,
)

SYSTEM_PROMPT = (
    "You are an expert video analyst specialised in driving scenes and "
    "AI-generation artifact detection. You will be given frames sampled at "
    "1 frame per second from a short driving clip, followed by a question. "
    "Analyse the frames carefully and return a structured answer: a 1-3 "
    "sentence rationale in `evaluation`, and the final choice in `answer` "
    "(which must match one of the allowed options exactly)."
)


def _frame_to_inline_part(frame: Frame) -> dict[str, Any]:
    raw = frame.path.read_bytes()
    return {
        "inline_data": {
            "mime_type": "image/jpeg",
            "data": base64.b64encode(raw).decode("ascii"),
        }
    }


def build_request_body(
    frames: list[Frame],
    question: str,
    schema_cls: type[_Base],
    *,
    reflection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one Gemini request body.

    `reflection` (optional) switches to the round-2 self-reflection prompt:
    pass the round-1 result record (needs `evaluation` / `answer`) and
    `contents` becomes
        user(images + question), model(round-1 JSON), user(re-examine)
    """
    options = allowed_answers(schema_cls)
    schema_json = schema_cls.model_json_schema()
    # Strict structured output requires `additionalProperties: false` on every
    # object; pydantic doesn't emit it by default.
    schema_json["additionalProperties"] = False

    question_text = (
        f"Question: {question}\n"
        f"Allowed answers: {options}\n"
        "Respond using the structured schema."
    )

    parts: list[dict[str, Any]] = [_frame_to_inline_part(f) for f in frames]
    parts.append({"text": question_text})

    contents: list[dict[str, Any]] = [{"role": "user", "parts": parts}]
    if reflection is not None:
        contents.append({
            "role": "model",
            "parts": [{"text": round1_assistant_text(
                reflection.get("evaluation", ""), reflection.get("answer", "")
            )}],
        })
        contents.append({"role": "user", "parts": [{"text": REFLECTION_PROMPT}]})

    return {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": contents,
        "generation_config": {
            "response_mime_type": "application/json",
            "response_json_schema": schema_json,
        },
    }


def batch_jsonl_line(custom_id: str, body: dict[str, Any]) -> dict[str, Any]:
    return {"key": custom_id, "request": body}
