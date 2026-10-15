"""ROC comparison on the same clips: anisotropy vs lattice score, raw vs n10.

  .venv/bin/python src/fft/roc_compare.py [--ids results/fft/e6_clip_ids.txt]
  -> report/figures/fft_roc_aniso_vs_lattice.{pdf,png}, results/fft/roc_compare.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, polars as pl
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]


def load(cond: str) -> pl.DataFrame:
    f = pl.read_parquet(REPO / f"results/fft/features_{cond}_f60.parquet").filter(pl.col("error").is_null()).unique("clip_id", keep="last")
    f = f.select("clip_id", pl.col("aniso_mean").alias("aniso"))
    lp = REPO / f"results/fft/lattice_{cond}_f4.parquet"
    if lp.is_file():
        l = pl.read_parquet(lp).filter(pl.col("error").is_null()).unique("clip_id", keep="last").select("clip_id", "lattice_score", "lattice_primary")
        f = f.join(l, on="clip_id", how="left")
    return f


def eer_threshold(y, x):
    fpr, tpr, thr = roc_curve(y, x); i = int(np.argmin(np.abs(tpr - (1 - fpr)))); return float(thr[i]), float(tpr[i]), float(fpr[i])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ids", default=str(REPO / "results/fft/e6_clip_ids.txt")); a = ap.parse_args()
    m = pl.read_parquet(REPO / "results/fft/manifest.parquet").select("clip_id", "label", "generator", "width", "height")
    ids = set(Path(a.ids).read_text().splitlines())
    data = {c: load(c).join(m, on="clip_id") for c in ("raw", "n10")}
    common = ids & set(data["raw"]["clip_id"]) & set(data["n10"]["clip_id"])
    out = {"n_common": len(common)}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    colors = {"aniso": "#1f77b4", "lattice_score": "#d62728", "lattice_primary": "#ff7f0e"}
    names = {"aniso": "anisotropy (published metric)", "lattice_score": "lattice score (36 sites)", "lattice_primary": "lattice primary (4 sites)"}
    for ax, cond in zip(axes, ("raw", "n10")):
        d = data[cond].filter(pl.col("clip_id").is_in(list(common)))
        y = (d["label"] == "generated").to_numpy().astype(int); out[cond] = {"n": int(d.height), "n_gen": int(y.sum()), "n_real": int((1 - y).sum())}
        for feat in ("aniso", "lattice_score", "lattice_primary"):
            if feat not in d.columns or d[feat].null_count() == d.height:
                continue
            dd = d.filter(pl.col(feat).is_not_null()); yy = (dd["label"] == "generated").to_numpy().astype(int); x = dd[feat].to_numpy()
            auc = roc_auc_score(yy, x); fpr, tpr, _ = roc_curve(yy, x); t, tp, fp = eer_threshold(yy, x)
            per_gen = {g: float(((dd.filter(pl.col("generator") == g)[feat].to_numpy()) >= t).mean()) for g in ("dd", "p1") if dd.filter(pl.col("generator") == g).height}
            out[cond][feat] = {"auc": round(float(auc), 4), "t_eer": round(t, 4), "tpr_eer": round(tp, 3), "fpr_eer": round(fp, 3), "tpr_per_generator_at_eer": per_gen,
                               "median_real": round(float(np.median(x[yy == 0])), 4), "median_gen": round(float(np.median(x[yy == 1])), 4)}
            ax.plot(fpr, tpr, color=colors[feat], lw=2, label=f"{names[feat]}  AUC={auc:.3f}")
        ax.plot([0, 1], [0, 1], ":", color="grey"); ax.set_xlabel("False positive rate (real called Generated)")
        ax.set_title({"raw": "raw files: real mpeg4/10fps vs generated h264/24fps", "n10": "n10: everything re-encoded h264 1280x704 10fps"}[cond] + f"\n(n={d.height}: {int(y.sum())} generated / {int((1-y).sum())} real)", fontsize=10)
        ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3)
    axes[0].set_ylabel("True positive rate (generated found)")
    fig.suptitle("Real vs generated: anisotropy vs lattice score, same clips", fontsize=12); fig.tight_layout()
    (REPO / "report/figures").mkdir(exist_ok=True)
    fig.savefig(REPO / "report/figures/fft_roc_aniso_vs_lattice.pdf"); fig.savefig(REPO / "report/figures/fft_roc_aniso_vs_lattice.png", dpi=140)
    json.dump(out, open(REPO / "results/fft/roc_compare.json", "w"), indent=2); print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
