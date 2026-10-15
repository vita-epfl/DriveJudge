"""Merge a redo pass into a second-annotator output directory.

  .venv/bin/python src/annotation/merge_annotator_outputs.py \
      --base results/annotation/annotator2 --override <dir with questions_*.json> \
      --match realistic --out results/annotation/annotator2_merged

For every clip in --override, answers whose question text contains --match replace the
same question's answer in --base (other answers untouched). Clips only in --override are added.
"""
import argparse, json, shutil
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--override", required=True)
ap.add_argument("--match", required=True, help="substring of the question text to take from --override")
ap.add_argument("--out", required=True)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
for f in sorted(Path(a.base).glob("questions_*.json")):
    base = json.load(open(f)); ov_path = Path(a.override) / f.name
    ov = {e["video"]: e for e in json.load(open(ov_path))} if ov_path.is_file() else {}
    replaced = added = 0
    by_video = {e["video"]: e for e in base}
    for vid, e in ov.items():
        new = [qa for qa in e.get("qa_pairs", []) if a.match in qa["question"]]
        if vid in by_video:
            kept = [qa for qa in by_video[vid].get("qa_pairs", []) if a.match not in qa["question"]]
            by_video[vid]["qa_pairs"] = kept + new; replaced += len(new)
        else:
            by_video[vid] = {"video": vid, "qa_pairs": new}; added += 1
    json.dump(list(by_video.values()), open(out / f.name, "w"), indent=2, ensure_ascii=False)
    print(f"{f.name}: {len(by_video)} clips, replaced {replaced} '{a.match}' answers, added {added} clips -> {out / f.name}")
