"""Lattice score: energy at the periodic grid peaks that diffusion/VAE decoders leave in the 2-D spectrum.

Generated clips (Cosmos) show bright dots at (i*W/8, j*H/8) -- strongest at (±W/4, ±H/4) -- in the
log-magnitude spectrum; real footage shows none.  The score is the mean background-subtracted
log-amplitude at those off-axis lattice sites, averaged over a few frames (the lattice is static
across frames, so 4 frames suffice).

Usage:
  .venv/bin/python src/fft/lattice.py --condition raw --frames 4 --workers 4 [--filter generator=p1]
  -> results/fft/lattice_<condition>_f<frames>.parquet  (clip_id, condition, lattice_score,
     lattice_peak, lattice_primary, lattice_secondary, n_sites, n_frames, width, height, error)
"""
from __future__ import annotations
import argparse, importlib.util, os, sys, time
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("compute_fft_agent", REPO / "src/agentic/fft/compute_fft.py")
cf = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(cf)

HARMONICS = [(i, j) for i in (-3, -2, -1, 1, 2, 3) for j in (-3, -2, -1, 1, 2, 3)]  # off-axis (i*W/8, j*H/8)
PRIMARY = {(2, 2), (2, -2), (-2, 2), (-2, -2)}   # (±W/4, ±H/4)


def lattice_from_spectrum(S: np.ndarray, win: int = 3, bg: int = 20) -> dict:
    """Per-site: peak = max of the (2*win+1)^2 centre patch, background = median of the (2*bg+1)^2 patch."""
    h, w = S.shape; cy, cx = h // 2, w // 2
    vals, prim, sec = [], [], []
    for i, j in HARMONICS:
        x = int(round(cx + i * w / 8)); y = int(round(cy + j * h / 8))
        if not (bg <= y < h - bg and bg <= x < w - bg):
            continue
        patch = S[y - bg:y + bg + 1, x - bg:x + bg + 1]
        v = float(S[y - win:y + win + 1, x - win:x + win + 1].max() - np.median(patch))
        vals.append(v); (prim if (i, j) in PRIMARY else sec).append(v)
    return {"lattice_score": float(np.mean(vals)), "lattice_peak": float(np.max(vals)),
            "lattice_primary": float(np.mean(prim)) if prim else float("nan"),
            "lattice_secondary": float(np.mean(sec)) if sec else float("nan"), "n_sites": len(vals)}


def lattice_features(path: str, n_frames: int = 4) -> dict:
    try:
        frames = cf.load_frames(Path(path), max_frames=n_frames)
        if len(frames) < 1:
            return {"error": "no frames"}
        S = np.mean([np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(f.astype(np.float32))))) for f in frames], axis=0)
        out = lattice_from_spectrum(S); out.update(n_frames=len(frames), width=S.shape[1], height=S.shape[0], error=None)
        return out
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


_SCHEMA = ("lattice_score", "lattice_peak", "lattice_primary", "lattice_secondary", "n_sites", "n_frames", "width", "height", "error")


def _work(args):
    cid, path, n = args
    t = time.time(); r = lattice_features(path, n)
    row = {k: r.get(k) for k in _SCHEMA}  # fixed schema so error rows don't break the parquet writer
    for k in ("lattice_score", "lattice_peak", "lattice_primary", "lattice_secondary"):
        row[k] = float(row[k]) if row[k] is not None else None
    for k in ("n_sites", "n_frames", "width", "height"):
        row[k] = int(row[k]) if row[k] is not None else None
    row.update(clip_id=cid, elapsed_s=round(time.time() - t, 3)); return row


def main():
    import polars as pl
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(REPO / "results/fft/manifest.parquet"))
    ap.add_argument("--condition", default="raw")
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--filter", default=None, help="key=value[,key=value]")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    m = pl.read_parquet(a.manifest)
    if "error" in m.columns:
        m = m.filter(pl.col("error").is_null())
    if a.filter:
        for kv in a.filter.split(","):
            k, v = kv.split("="); v = {"true": True, "false": False}.get(v.lower(), v)
            m = m.filter(pl.col(k) == v)
    if a.condition != "raw":
        root = REPO / "dataset/fft_conditions" / a.condition
        m = m.with_columns((pl.lit(str(root) + "/") + pl.col("clip_id") + ".mp4").alias("path"))
        m = m.filter(pl.col("path").map_elements(lambda p: Path(p).is_file(), return_dtype=pl.Boolean))
    if a.limit:
        m = m.head(a.limit)
    out = Path(a.out or REPO / f"results/fft/lattice_{a.condition}_f{a.frames}.parquet")
    done = set()
    if out.is_file():
        done = set(pl.read_parquet(out)["clip_id"].to_list())
    todo = [(r["clip_id"], r["path"], a.frames) for r in m.iter_rows(named=True) if r["clip_id"] not in done]
    print(f"condition={a.condition} frames={a.frames} clips={m.height} todo={len(todo)} workers={a.workers}", flush=True)
    rows = []; t0 = time.time()
    with Pool(a.workers, maxtasksperchild=100) as pool:
        for k, r in enumerate(pool.imap_unordered(_work, todo, chunksize=4), 1):
            rows.append(r)
            if k % 200 == 0 or k == len(todo):
                el = time.time() - t0
                print(f"  {k}/{len(todo)}  {el/60:.1f} min  {k/el:.2f} clip/s", flush=True)
                df = pl.DataFrame(rows, schema_overrides={'error': pl.Utf8, 'lattice_score': pl.Float64, 'lattice_peak': pl.Float64, 'lattice_primary': pl.Float64, 'lattice_secondary': pl.Float64}).with_columns(pl.lit(a.condition).alias("condition"))
                if out.is_file():
                    df = pl.concat([pl.read_parquet(out), df], how="diagonal_relaxed")
                tmp = out.with_suffix(".tmp.parquet"); df.write_parquet(tmp); tmp.replace(out); rows = []
    print("done", out)


if __name__ == "__main__":
    main()
