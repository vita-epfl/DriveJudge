"""Offline tests for src/agentic/video_resolve.py (plain asserts, no pytest).

    .venv/bin/python src/agentic/counterfactual/test_resolver.py

Builds a temporary ``final_dataset``-shaped tree in which the same stem exists
in two pools, then checks that a bare filename resolves to the *current item's*
copy, that relative and absolute identifiers still work, and that an ambiguous
basename with no current item warns instead of resolving silently.

Imports ``video_resolve`` directly, so qwen-agent / torch / a GPU are not
needed.
"""

from __future__ import annotations

import logging
import shutil
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import video_resolve as vr  # noqa: E402


class _CaptureWarnings(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record.getMessage())


def _make_tree(root: Path) -> dict[str, Path]:
    """final_dataset/{cosmos-drive-dreams,cosmos-predict1,cosmos-predict2}/..."""
    paths = {
        "dd": root / "cosmos-drive-dreams" / "negative" / "car_brake_001.mp4",
        "p1": root / "cosmos-predict1" / "artifacts" / "car_brake_001.mp4",
        "p2": root / "cosmos-predict2" / "car_brake" / "car_brake_001.mp4",
        "unique": root / "cosmos-drive-dreams" / "negative" / "agg_tk_030_Foggy.mp4",
    }
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\x00fake mp4\x00")
    return paths


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="cf_resolver_test_"))
    handler = _CaptureWarnings()
    vr.logger.addHandler(handler)
    vr.logger.setLevel(logging.WARNING)
    try:
        root = tmp / "final_dataset"
        paths = _make_tree(root)

        cache = vr.configure_resolver(root, legacy_fix=None)
        assert len(cache["car_brake_001.mp4"]) == 3, cache["car_brake_001.mp4"]
        assert len(cache["agg_tk_030_Foggy.mp4"]) == 1
        print(f"[1] cache: {len(cache)} basenames, "
              f"{sum(1 for v in cache.values() if len(v) > 1)} ambiguous -- OK")

        # ---- 2. bare filename resolves to the CURRENT item, in every pool ----
        for tag in ("dd", "p1", "p2"):
            vr.set_current_item(paths[tag])
            got = vr.get_absolute_path("car_brake_001.mp4")
            assert got == paths[tag], f"{tag}: expected {paths[tag]}, got {got}"
        vr.pop_resolution_log()
        assert not handler.records, f"unexpected warnings: {handler.records}"
        print("[2] bare basename -> the current item's own copy in dd / p1 / p2 -- OK")

        # ---- 3. the OLD behaviour would have been wrong ----
        old_cache = {p.name: p for p in sorted(root.rglob("*.mp4"))}  # last writer wins
        old_pick = old_cache["car_brake_001.mp4"]
        assert sum(old_pick == paths[t] for t in ("dd", "p1", "p2")) == 1
        wrong = [t for t in ("dd", "p1", "p2") if paths[t] != old_pick]
        assert len(wrong) == 2
        print(f"[3] old basename cache always returned {old_pick.parts[-3]} "
              f"-> wrong for {wrong} -- reproduced")

        # ---- 4. relative and absolute identifiers still work ----
        vr.set_current_item(paths["dd"])
        rel = "cosmos-predict1/artifacts/car_brake_001.mp4"
        assert vr.get_absolute_path(rel) == paths["p1"], "relative identifier"
        assert vr.get_absolute_path(str(paths["p2"])) == paths["p2"], "absolute identifier"
        # an explicit path must win over the current item even though the
        # basenames match
        assert vr.get_absolute_path(str(paths["p1"])) == paths["p1"]
        assert not handler.records, f"unexpected warnings: {handler.records}"
        print("[4] relative + absolute identifiers unchanged (and beat the current item) -- OK")

        # ---- 5. a unique basename resolves from the cache, no warning ----
        vr.set_current_item(paths["p1"])
        assert vr.get_absolute_path("agg_tk_030_Foggy.mp4") == paths["unique"]
        assert not handler.records
        print("[5] unique basename -> cache hit, no warning -- OK")

        # ---- 6. ambiguity with no current item: WARN + first sorted ----
        vr.clear_current_item()
        handler.records.clear()
        got = vr.get_absolute_path("car_brake_001.mp4")
        assert len(handler.records) == 1, handler.records
        msg = handler.records[0]
        assert "Ambiguous video basename" in msg
        for tag in ("dd", "p1", "p2"):
            assert str(paths[tag]) in msg, f"candidate {tag} missing from the warning"
        assert got == sorted(paths[t] for t in ("dd", "p1", "p2"))[0], got
        events = vr.pop_resolution_log()
        assert len(events) == 1 and events[0]["rule"] == "first_sorted", events
        assert len(events[0]["candidates"]) == 3
        print("[6] ambiguous basename, no current item -> WARNING + first sorted -- OK")

        # ---- 7. ambiguity whose current item lives elsewhere: same-pool wins --
        other = root / "cosmos-predict2" / "car_brake" / "other_clip.mp4"
        other.write_bytes(b"\x00fake mp4\x00")
        vr.configure_resolver(root, legacy_fix=None)
        vr.set_current_item(other)
        handler.records.clear()
        got = vr.get_absolute_path("car_brake_001.mp4")
        assert got == paths["p2"], got
        assert len(handler.records) == 1 and "same pool" in handler.records[0]
        events = vr.pop_resolution_log()
        assert events[0]["rule"] == "same_pool" and events[0]["current_pool"] == "cosmos-predict2"
        print("[7] ambiguous basename, current item in another clip of the same pool "
              "-> WARNING + same-pool candidate -- OK")

        # ---- 8. legacy /mnt/vita/scratch mount fix still applied ----
        vr.configure_resolver(root, legacy_fix=("/NOT_MOUNTED", str(root)))
        vr.set_current_item("/NOT_MOUNTED/cosmos-drive-dreams/negative/car_brake_001.mp4")
        assert vr.current_item() == paths["dd"]
        assert vr.get_absolute_path(
            "/NOT_MOUNTED/cosmos-predict1/artifacts/car_brake_001.mp4") == paths["p1"]
        print("[8] legacy mount rewrite applied to both the item and the tool argument -- OK")

        # ---- 9. the context is per-thread ----
        vr.configure_resolver(root, legacy_fix=None)
        vr.set_current_item(paths["dd"])
        seen: dict[str, Path] = {}

        def worker(tag: str) -> None:
            vr.set_current_item(paths[tag])
            seen[tag] = vr.get_absolute_path("car_brake_001.mp4")

        threads = [threading.Thread(target=worker, args=(t,)) for t in ("p1", "p2")]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert seen == {"p1": paths["p1"], "p2": paths["p2"]}, seen
        assert vr.get_absolute_path("car_brake_001.mp4") == paths["dd"], "main thread clobbered"
        print("[9] threading.local context is per-worker -- OK")

        # ---- 10. an unknown basename still raises ----
        try:
            vr.get_absolute_path("does_not_exist.mp4")
        except FileNotFoundError:
            print("[10] unknown basename -> FileNotFoundError -- OK")
        else:
            raise AssertionError("expected FileNotFoundError")

        print("\nAll resolver tests passed.")
        return 0
    finally:
        vr.logger.removeHandler(handler)
        vr.clear_current_item()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
