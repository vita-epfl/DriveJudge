"""Map cluster video paths in Questions_raw.json to local files.

Questions_raw.json stores paths like
    /mnt/vita/scratch/vita-students/users/scharffe/VLM-eval/final_dataset/cosmos-drive-dreams/…
which only exist on the RCP cluster. Locally the same tree lives under
    <LOCAL_DATASET_ROOT>/cosmos-drive-dreams/…
and the real clips (`final_dataset/real/NNNNNN.mp4` on the cluster) live under
    <LOCAL_REAL_ROOT>/NNNNNN.mp4   (CarCrash "Normal" split)

Downstream analysis (`src/analysis/answer_analysis_agent.py`) strips the cluster
prefix from `el["video"]` to get the relative key (e.g.
`cosmos-drive-dreams/negative/agg_tk_030_Foggy.mp4`). We therefore keep the
cluster path in the saved results and only rewrite to the local absolute path
when we actually need to open the video.

Environment overrides:
    LOCAL_DATASET_ROOT   local copy of final_dataset (default: ../final_dataset
                         next to the repo checkout)
    LOCAL_REAL_ROOT      folder holding the real clips (default:
                         ../Dataset/CarCrash/videos/Normal next to the repo)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

CLUSTER_PREFIX = "/mnt/vita/scratch/vita-students/users/scharffe/VLM-eval/final_dataset/"

_HERE = Path(__file__).resolve().parent
# <repo>/src/evaluation/closed_source/gpt -> parents[3] = <repo>, parents[4] = its parent dir
_PROJECT_DIR = _HERE.parents[4]
_DEFAULT_LOCAL_ROOT = _PROJECT_DIR / "final_dataset"
_DEFAULT_REAL_ROOT = _PROJECT_DIR / "Dataset" / "CarCrash" / "videos" / "Normal"

# Sub-trees of final_dataset whose files are not generated clips and live in a
# separate local folder. Key = first path segment after the cluster prefix.
_REAL_SUBDIRS = {"real"}


def local_dataset_root() -> Path:
    root = os.environ.get("LOCAL_DATASET_ROOT")
    return Path(root).resolve() if root else _DEFAULT_LOCAL_ROOT


def local_real_root() -> Path:
    root = os.environ.get("LOCAL_REAL_ROOT")
    return Path(root).resolve() if root else _DEFAULT_REAL_ROOT


@dataclass(frozen=True)
class QuestionItem:
    cluster_path: str      # unchanged `/mnt/.../final_dataset/...` — saved in results
    relative_path: str     # `cosmos-drive-dreams/...` or `real/...` — join key with ground-truth
    local_path: Path       # absolute path on this machine
    question: str          # question text, exactly as stored in Questions_raw.json


def _cluster_to_local(cluster_path: str, root: Path) -> tuple[str, Path]:
    if not cluster_path.startswith(CLUSTER_PREFIX):
        raise ValueError(f"Unexpected video path prefix: {cluster_path!r}")
    relative = cluster_path[len(CLUSTER_PREFIX):]
    first, _, rest = relative.partition("/")
    if first in _REAL_SUBDIRS and rest:
        return relative, (local_real_root() / rest).resolve()
    return relative, (root / relative).resolve()


def load_questions(
    questions_raw_path: Path | str,
    *,
    require_local_exists: bool = True,
) -> list[QuestionItem]:
    with open(questions_raw_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    root = local_dataset_root()
    items: list[QuestionItem] = []
    missing: list[str] = []

    for entry in raw:
        # Each entry is a chat-formatted user message with [video, text].
        content = entry["content"]
        video = next(c for c in content if c["type"] == "video")["video"]
        text = next(c for c in content if c["type"] == "text")["text"]
        rel, local = _cluster_to_local(video, root)
        if require_local_exists and not local.is_file():
            missing.append(str(local))
            continue
        items.append(
            QuestionItem(
                cluster_path=video,
                relative_path=rel,
                local_path=local,
                question=text,
            )
        )

    if missing:
        print(f"[path_fix] Skipped {len(missing)} items whose local video is missing.")
        print(f"[path_fix] First missing: {missing[0]}")

    return items


def iter_questions(
    questions_raw_path: Path | str,
    *,
    limit: int | None = None,
) -> Iterator[QuestionItem]:
    for i, item in enumerate(load_questions(questions_raw_path)):
        if limit is not None and i >= limit:
            return
        yield item


if __name__ == "__main__":
    # Tiny sanity check — prints counts, does not dump all items.
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else str(_HERE.parents[3] / "dataset/dataset_final/Questions_raw.json")
    print(f"[path_fix] dataset root: {local_dataset_root()}  (exists={local_dataset_root().is_dir()})")
    print(f"[path_fix] real root:    {local_real_root()}  (exists={local_real_root().is_dir()})")
    items = load_questions(path)
    by_video: dict[str, int] = {}
    for it in items:
        by_video[it.relative_path] = by_video.get(it.relative_path, 0) + 1
    print(f"{len(items)} (video, question) pairs resolved locally.")
    print(f"{len(by_video)} unique videos.")
    print(f"Mean questions/video: {len(items) / max(1, len(by_video)):.2f}")
