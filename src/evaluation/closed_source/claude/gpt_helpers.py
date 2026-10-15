"""Shim that exposes the gpt/ helpers from the sibling folder.

`path_fix`, `frame_extractor`, `schemas` and `prompt_builder` are dataset /
prompt helpers that are identical for the Claude run, so we import them in
place (from `src/evaluation/closed_source/gpt/`) rather than copy. Reusing
`SYSTEM_PROMPT` and `REFLECTION_PROMPT` verbatim is what keeps the Claude arm
comparable with the GPT and Gemini arms.
"""

from __future__ import annotations

import sys
from pathlib import Path

_GPT_DIR = (Path(__file__).resolve().parent.parent / "gpt").resolve()
if str(_GPT_DIR) not in sys.path:
    sys.path.insert(0, str(_GPT_DIR))

from path_fix import QuestionItem, load_questions  # noqa: E402,F401
from frame_extractor import Frame, extract_frames  # noqa: E402,F401
from schemas import (  # noqa: E402,F401
    _Base,
    allowed_answers,
    schema_for,
)
from prompt_builder import (  # noqa: E402,F401
    REFLECTION_PROMPT,
    SYSTEM_PROMPT,
    build_question_text,
    round1_assistant_text,
)
from naming import output_stem, sanitise_model  # noqa: E402,F401
