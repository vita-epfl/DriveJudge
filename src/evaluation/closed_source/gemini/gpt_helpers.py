"""Shim that exposes the gpt/ helpers from the sibling folder.

`path_fix`, `frame_extractor`, and `schemas` are dataset/format helpers that are
identical for the Gemini run, so we import them in place rather than copy.
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
