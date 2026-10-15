"""Merge result JSONs, deduplicating by custom_id — thin wrapper over gpt/.

The merge logic is backend-agnostic (dedupe by `custom_id`, last file wins), so
rather than fork it we load `gpt/merge_results.py` by path and call its `main`.
Loading by path avoids the name clash a plain `import merge_results` would hit.

Usage:
    python merge_results.py \\
        results/claude_batch_results.json \\
        results/claude_live_results.json \\
        -o results/claude_all_results.json
"""

from __future__ import annotations

import importlib.util
import sys

from gpt_helpers import _GPT_DIR

_SPEC = importlib.util.spec_from_file_location(
    "gpt_merge_results", _GPT_DIR / "merge_results.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_gpt_merge = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_gpt_merge)

main = _gpt_merge.main

if __name__ == "__main__":
    sys.exit(main())
