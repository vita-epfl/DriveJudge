"""ROC of one detector on the same clips under two conditions (raw vs n10), with scenario-bootstrap CI.

  .venv/bin/python src/fft/roc_two_conditions.py --feature aniso_mean   --threshold 0.048
  .venv/bin/python src/fft/roc_two_conditions.py --feature lattice_primary
  -> report/figures/fft_roc_raw_vs_n10_<feature>.{pdf,png}, results/fft/roc_raw_vs_n10_<feature>.json
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np, polars as pl
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
LABELS = {"raw": "original files (real mpeg4 10 fps, generated h264 24 fps)", "n10": "same clips re-encoded to one container (h264, 1280×704, 10 fps)"}
NAMES = {"aniso_mean": "FFT anisotropy (published metric)", "lattice_primary": "FFT lattice score (4 primary sites)", "lattice_score": "FFT lattice score (36 sites)"}


def scenario_id(clip_id: str) -> str:
    stem = re.sub(r"_(Original|Morning|Night|Snowy|Rainy|Foggy|Sunny|Golden hour)$", "", clip_id)
    return re.sub(r"_(7b|14b)$", "", stem)


def load(cond: str, feature: str, ids: set[str]) -> pl.DataFrame:
    if feature.startswith("lattice"):
        f = pl.read_parquet(REPO / f"results/fft/lattice_{cond}_f4.parquet")
    else:
        f = pl.read_parquet(REPO / f"results/fft/features_{cond}_f60.parquet")
    f = f.filter(pl.col("error").is_null()).unique("clip_id", keep="last").select("clip_id", feature)
    return f.filter(pl.col("clip_id").is_in(list(ids)))


def boot_auc(y, x, groups, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed); g = np.asarray(groups); ug, inv = np.unique(g, return_inverse=True); C = len(ug)
    aucs = []
    for _ in range(n_boot):
        w = rng.multinomial(C, np.full(C, 1 / C))[inv]  # cluster weights
        if (w[y == 1]).sum() == 0 or (w[y == 0]).sum() == 0: continue
        aucs.append(roc_auc_score(y, x, sample_weight=w))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--feature", default="aniso_mean"); ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--ids", default=str(REPO / "results/fft/e6_clip_ids.txt")); ap.add_argument("--n-boot", type=int, default=2000); a = ap.parse_args()
    m = pl.read_parquet(REPO / "results/fft/manifest.parquet").select("clip_id", "label", "generator")
    ids = set(Path(a.ids).read_text().splitlines()); out = {"feature": a.feature}
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    for cond, col in (("raw", "#1f77b4"), ("n10", "#d62728")):
        d = load(cond, a.feature, ids).join(m, on="clip_id")
        y = (d["label"] == "generated").to_numpy().astype(int); x = d[a.feature].to_numpy(); grp = [scenario_id(c) for c in d["clip_id"]]
        fpr, tpr, thr = roc_curve(y, x); auc = roc_auc_score(y, x); lo, hi = boot_auc(y, x, grp, a.n_boot)
        ax.plot(fpr, tpr, color=col, lw=2.2, label=f"{LABELS[cond]}\nAUC = {auc:.3f}, 95 % CI [{lo:.3f}, {hi:.3f}]")
        rec = {"n": int(d.height), "n_gen": int(y.sum()), "auc": round(float(auc), 4), "ci": [round(lo, 4), round(hi, 4)],
               "median_real": round(float(np.median(x[y == 0])), 4), "median_gen": round(float(np.median(x[y == 1])), 4)}
        i = int(np.argmin(np.abs(tpr - (1 - fpr)))); rec["t_eer"] = round(float(thr[i]), 4); rec["tpr_eer"] = round(float(tpr[i]), 3); rec["fpr_eer"] = round(float(fpr[i]), 3)
        t = a.threshold if a.threshold is not None else rec["t_eer"]
        fp, tp = float((x[y == 0] >= t).mean()), float((x[y == 1] >= t).mean()); rec[f"tpr_at_{t}"] = round(tp, 3); rec[f"fpr_at_{t}"] = round(fp, 3)
        ax.plot([fp], [tp], "o", color=col, ms=7, mec="k"); ax.annotate(f"t = {t:.3g}", (fp, tp), xytext=(8, -12), textcoords="offset points", fontsize=8, color=col)
        for g in ("dd", "p1"):
            sub = d.filter(pl.col("generator") == g)
            if sub.height: rec[f"tpr_{g}_at_{t}"] = round(float((sub[a.feature].to_numpy() >= t).mean()), 3)
        out[cond] = rec
    ax.plot([0, 1], [0, 1], ":", color="grey"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.001)
    ax.set_xlabel("False positive rate (real clips called Generated)"); ax.set_ylabel("True positive rate (generated clips found)")
    ax.set_title(f"{NAMES.get(a.feature, a.feature)}: 1000 real vs 1000 generated clips", fontsize=11)
    ax.legend(loc="lower right", fontsize=8.5, frameon=True, title=f"CI: bootstrap over scenarios ({a.n_boot} resamples)", title_fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
    stem = REPO / f"report/figures/fft_roc_raw_vs_n10_{a.feature}"; fig.savefig(stem.with_suffix(".pdf")); fig.savefig(stem.with_suffix(".png"), dpi=150)
    json.dump(out, open(REPO / f"results/fft/roc_raw_vs_n10_{a.feature}.json", "w"), indent=2); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
