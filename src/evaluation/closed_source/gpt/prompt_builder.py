"""Build chat messages with a cacheable prefix.

Layout (ordered):
    system      : constant across *every* request          ─┐
    user[0..N-1]: the N frames of the video                 │ cached prefix
                                                            ┘
    user[N]     : question text + allowed answers           — variable suffix

OpenAI prompt-caching is prefix-based: keeping image tokens identical across
requests for the same video, and putting them *before* the question text, is
the whole point. We pack the frames in one user message followed by the
question text in the same user message's content list — that preserves token
order `system → images → text` which is what the cache keys on.
"""

from __future__ import annotations

from typing import Literal

from frame_extractor import Frame
from schemas import _Base, allowed_answers

ImageDetail = Literal["low", "high", "auto"]

SYSTEM_PROMPT = (
    "You are an expert video analyst specialised in driving scenes and "
    "AI-generation artifact detection. You will be given frames sampled at "
    "1 frame per second from a short driving clip, followed by a question. "
    "Analyse the frames carefully and return a structured answer: a 1–3 "
    "sentence rationale in `evaluation`, and the final choice in `answer` "
    "(which must match one of the allowed options exactly)."
)

# Round-2 user turn of the self-reflection variant. Shared verbatim by the GPT,
# Gemini and Claude backends so the three arms stay comparable.
REFLECTION_PROMPT = (
    "Re-examine the frames. Check each claim of your previous rationale "
    "against what is visible, consider the opposite answer, then return a "
    "revised structured answer in the same schema (it may be unchanged)."
)


def build_question_text(question: str, schema: type[_Base]) -> str:
    """The variable suffix of the prompt — identical across all three backends."""
    return (
        f"Question: {question}\n"
        f"Allowed answers: {allowed_answers(schema)}\n"
        "Respond using the structured schema."
    )


def round1_assistant_text(evaluation: str, answer: str) -> str:
    """Serialise a round-1 answer back into the exact JSON the model emitted.

    Key order is fixed (`evaluation`, then `answer`) so the round-2 prompt is
    byte-stable across reruns — a varying prefix would kill prompt caching.
    """
    import json

    return json.dumps({"evaluation": evaluation, "answer": answer},
                      ensure_ascii=False)


def build_messages(
    frames: list[Frame],
    question: str,
    schema: type[_Base],
    *,
    image_detail: ImageDetail = "low",
    reflection: dict | None = None,
) -> list[dict]:
    """Build the chat messages.

    `reflection` (optional) turns this into the round-2 self-reflection prompt:
    pass the round-1 result record (needs `evaluation` and `answer`) and the
    returned message list becomes
        system, user(images + question), assistant(round-1 JSON), user(re-examine)
    Everything before the assistant turn is byte-identical to round 1, so the
    cached image prefix is reused.
    """
    image_parts = [
        {
            "type": "image_url",
            "image_url": {"url": f.data_url, "detail": image_detail},
        }
        for f in frames
    ]
    question_text = build_question_text(question, schema)
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                *image_parts,
                {"type": "text", "text": question_text},
            ],
        },
    ]
    if reflection is not None:
        messages.append({
            "role": "assistant",
            "content": round1_assistant_text(
                reflection.get("evaluation", ""), reflection.get("answer", "")
            ),
        })
        messages.append({"role": "user", "content": REFLECTION_PROMPT})
    return messages
