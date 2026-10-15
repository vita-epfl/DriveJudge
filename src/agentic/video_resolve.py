"""Provenance-safe video-path resolution for the DriveJudgeBench agents.

Why this module exists
----------------------
The v4 agent used to keep a ``basename -> Path`` cache built by ``rglob``-ing
``final_dataset``::

    _VIDEO_CACHE[video_path.name] = video_path      # last writer wins

The VLM hands the tools a *bare filename* (verified on the v4o run: 3158/3158
tool calls passed a basename, never a path), and 190 mp4 basenames exist in more
than one pool of ``final_dataset`` (``cosmos-drive-dreams`` /
``cosmos-predict1`` / ``cosmos-predict2`` -- e.g. ``car_brake_001.mp4`` appears
three times).  For those clips the cache silently resolved to whichever copy
``rglob`` happened to visit last, so RAFT / SAM / FFT could be run on a
*different video than the one the question was about*.

This module fixes that by resolving against the item currently being processed
by the calling worker thread:

1. an absolute path that exists is used as-is,
2. a relative path is resolved under ``ROOT_DIR``,
3. a bare filename equal to the current item's filename resolves to the current
   item's own path (this is the case that used to be wrong),
4. otherwise the basename cache is consulted; the cache maps a basename to the
   *list* of every matching file, and an ambiguous basename is never resolved
   silently -- a WARNING naming all candidates is logged and the candidate in
   the same top-level source dir as the current item wins, else the first
   sorted candidate.

The module deliberately imports nothing beyond the standard library so it can be
unit-tested (``counterfactual/test_resolver.py``) without qwen-agent, torch or a
GPU.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

__all__ = [
    "LEGACY_MOUNT_FIX",
    "build_video_cache",
    "clear_current_item",
    "configure_resolver",
    "current_item",
    "get_absolute_path",
    "pop_resolution_log",
    "set_current_item",
]

logger = logging.getLogger("video_resolve")

#: ``Questions_raw.json`` was written against ``/mnt/vita/scratch/...`` while the
#: canonical pod mounts the PVC at ``/mnt/vita/...``.  Set to ``None`` via
#: :func:`configure_resolver` on clusters that really mount ``scratch``.
LEGACY_MOUNT_FIX: tuple[str, str] | None = (
    "/mnt/vita/scratch/vita-students",
    "/mnt/vita/vita-students",
)

_ROOT_DIR: Path | None = None
_VIDEO_CACHE: dict[str, list[Path]] = {}
_legacy_fix: tuple[str, str] | None = LEGACY_MOUNT_FIX
_local = threading.local()


# ---------------------------------------------------------------------------
# cache construction
# ---------------------------------------------------------------------------


def build_video_cache(root: Path | str) -> dict[str, list[Path]]:
    """Map every mp4 basename under ``root`` to the **list** of matching files.

    Unlike the old ``dict[str, Path]`` cache this keeps every homonym, which is
    what makes the ambiguity detectable at resolution time.
    """
    root = Path(root)
    cache: dict[str, list[Path]] = {}
    for ext in ("*.mp4", "*.MP4"):
        for video_path in root.rglob(ext):
            if video_path.is_file():
                cache.setdefault(video_path.name, []).append(video_path)
    for paths in cache.values():
        paths.sort()
    return cache


def configure_resolver(
    root: Path | str,
    legacy_fix: tuple[str, str] | None = LEGACY_MOUNT_FIX,
) -> dict[str, list[Path]]:
    """Point the resolver at ``root`` and (re)build the basename cache."""
    global _ROOT_DIR, _VIDEO_CACHE, _legacy_fix
    _ROOT_DIR = Path(root)
    _legacy_fix = legacy_fix
    _VIDEO_CACHE = build_video_cache(_ROOT_DIR)
    return _VIDEO_CACHE


# ---------------------------------------------------------------------------
# per-worker "current item" context
# ---------------------------------------------------------------------------


def set_current_item(video_path: Path | str | None) -> None:
    """Pin the calling thread to the video of the item it is working on.

    Called once per item, right before the agent is run, from the item's own
    ``{"type": "video", "video": <abs path>}`` entry in the questions JSON.
    """
    if video_path is None:
        _local.current = None
        return
    p = Path(_apply_legacy_fix(str(video_path)))
    if not p.is_absolute() and _ROOT_DIR is not None:
        candidate = _ROOT_DIR / p
        if candidate.is_file():
            p = candidate
    _local.current = p


def clear_current_item() -> None:
    _local.current = None


def current_item() -> Path | None:
    return getattr(_local, "current", None)


def pop_resolution_log() -> list[dict]:
    """Return (and reset) this thread's list of ambiguous-resolution events."""
    events = getattr(_local, "log", None) or []
    _local.log = []
    return list(events)


def _record(event: dict) -> None:
    log = getattr(_local, "log", None)
    if log is None:
        log = _local.log = []
    log.append(event)


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------


def _apply_legacy_fix(identifier: str) -> str:
    if _legacy_fix:
        return identifier.replace(_legacy_fix[0], _legacy_fix[1])
    return identifier


def _pool_of(path: Path) -> str | None:
    """Top-level source dir of ``path`` relative to ``ROOT_DIR`` (the "pool")."""
    if _ROOT_DIR is None:
        return None
    try:
        return Path(path).resolve().relative_to(Path(_ROOT_DIR).resolve()).parts[0]
    except (ValueError, IndexError, OSError):
        return None


def get_absolute_path(identifier: str) -> Path:
    """Resolve a tool's ``video_path`` argument to an absolute file path."""
    identifier = _apply_legacy_fix(str(identifier).strip())
    ident_path = Path(identifier)

    # 1. an absolute (or cwd-relative) path that really exists
    if ident_path.exists() and ident_path.is_file():
        return ident_path

    # 2. something that looks like a path -> resolve under the dataset root
    if ("/" in identifier or "\\" in identifier) and _ROOT_DIR is not None:
        video_path = _ROOT_DIR / identifier
        if video_path.is_file():
            return video_path

    # 3. a bare filename that is the current item's own file  <-- the fix
    current = current_item()
    if current is not None and ident_path.name == current.name and current.is_file():
        return current

    # 4. fall back to the basename cache, never silently on an ambiguity
    candidates = _VIDEO_CACHE.get(ident_path.name)
    if candidates:
        if len(candidates) == 1:
            return candidates[0]
        wanted_pool = _pool_of(current) if current is not None else None
        same_pool = [p for p in candidates if _pool_of(p) == wanted_pool] if wanted_pool else []
        chosen = same_pool[0] if same_pool else sorted(candidates)[0]
        logger.warning(
            "Ambiguous video basename %r: %d candidates %s; current item = %s "
            "(pool %s) -> picked %s (%s)",
            ident_path.name,
            len(candidates),
            [str(p) for p in candidates],
            current,
            wanted_pool,
            chosen,
            "same pool as current item" if same_pool else "first sorted candidate",
        )
        _record(
            {
                "identifier": ident_path.name,
                "candidates": [str(p) for p in candidates],
                "current_item": str(current) if current is not None else None,
                "current_pool": wanted_pool,
                "chosen": str(chosen),
                "rule": "same_pool" if same_pool else "first_sorted",
            }
        )
        return chosen

    raise FileNotFoundError(f"Video '{identifier}' not found in {_ROOT_DIR}")
