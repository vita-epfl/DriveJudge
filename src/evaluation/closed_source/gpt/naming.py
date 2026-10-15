"""Model-id → output-filename helpers, shared by the gpt / gemini / claude runs.

Every backend used to hardcode its result filenames (`gpt_batch_results.json`,
`gemini_batch_results.json`, …), which silently collides as soon as a second
model id is evaluated with the same script. `output_stem` keeps the historical
name while the backend's *default* model is in use — so nothing already on disk
breaks — and switches to the sanitised model id for anything else.

    output_stem("gpt-5.4-mini", default_model="gpt-5.4-mini", default_stem="gpt")
        -> "gpt"                      (→ results/gpt_batch_results.json)
    output_stem("gpt-5.5",      default_model="gpt-5.4-mini", default_stem="gpt")
        -> "gpt-5.5"                  (→ results/gpt-5.5_batch_results.json)
"""

from __future__ import annotations

import re

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitise_model(model: str) -> str:
    """Filesystem-safe version of a model id (`/`, spaces, `:` → `_`)."""
    cleaned = _UNSAFE.sub("_", (model or "").strip()).strip("._-")
    return cleaned or "model"


def output_stem(model: str, *, default_model: str, default_stem: str) -> str:
    """`default_stem` for the backend's default model, sanitised id otherwise."""
    return default_stem if model == default_model else sanitise_model(model)
