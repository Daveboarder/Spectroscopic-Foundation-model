"""
Measured-vs-synthetic spectra diagnostic: PCA + k-means on random shots of a
LIBS raster, cluster mean spectra overlaid on synthetic mineral spectra.

Question it answers: do the synthetic spectra look like the measured ones at
all?  If the synthetic minerals fall outside the measured clusters in PCA
space, the generator (not the classifier) is the first thing to fix.

Steps
  1. ``--n`` random valid shots of the HDF5 map (``Invalid == 0``).
  2. Both axes are stitched to a monotonic axis (overlapping spectrometer
     channels: each channel continues where the previous one ended), the
     synthetic spectra are interpolated onto the measured axis over the common
     range, and every spectrum is unit-normalised (``libs_pipeline.unit_norm``,
     the training convention).
  3. PCA (``--n_pca`` components) on the measured spectra, k-means
     (``--k`` clusters) on the PCA scores; synthetic spectra are projected
     with the same PCA.
Outputs (``Outputs/pca_kmeans_<h5 stem>/``): ``spectra.html`` (class means +
synthetic mineral means, linear / log toggle), ``pca.html`` (PC scatter with
projected synthetic spectra and the shot positions coloured by cluster),
``summary.json``.

Usage:
    uv run python scripts/compare_measured_synthetic_pca.py \\
        --h5 .../Data/MAR1A/LIBS/Mar1A.h5 \\
        --synthetic external_data/cache/synthetic_cache_502e294c8785.h5 \\
        --minerals Albite Quartz Muscovite Spessartine "Biotite (Fe-rich, annite)"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.libs_pipeline import load_wavelength, unit_norm  # noqa: E402
from publication.inference_runner import read_sample_table  # noqa: E402

SATURATION_COUNTS = 64000.0  # Avantes 16-bit ceiling seen in Mar1A (~64,800)


def monotonic_mask(wl: np.ndarray) -> np.ndarray:
    """Keep pixel i when wl[i] exceeds every kept pixel before it (drops the
    re-measured start of each overlapping channel)."""
    keep = np.zeros(wl.size, dtype=bool)
    last = -np.inf
    for i, w in enumerate(wl):
        if w > last:
            keep[i] = True
            last = w
    return keep


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--h5", required=True)
    ap.add_argument("--measurement", default=None)
    ap.add_argument("--synthetic", required=True, help="synthetic_cache_*.h5")
    ap.add_argument(
        "--synthetic_axis",
        default="external_data/Data/matrix_01.h5",
        help="wavelength file of the synthetic cache (paths.wavelength_json of its data config)",
    )
    ap.add_argument("--minerals", nargs="+", required=True, help="sample_type_name values")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--k", type=int, default=9)
    ap.add_argument("--n_pca", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    h5_path = Path(args.h5)
    out_dir = Path(args.out_dir or ROOT / "Outputs" / f"pca_kmeans_{h5_path.stem}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ── measured ──
    with h5py.File(h5_path, "r") as f:
        key = args.measurement or sorted(f["measurements"])[0]
        g = f["measurements"][key]["libs"]
        wl_m = g["calibration"][...].astype(np.float64)
        md = {k: g["metadata"][k][...] for k in ("X", "Y", "X_pos", "Y_pos", "Invalid")}
        pool = np.nonzero(md["Invalid"] == 0)[0]
        idx = np.sort(rng.choice(pool, size=min(args.n, pool.size), replace=False))
        print(f"reading {idx.size} random shots of {pool.size:,} valid ...", flush=True)
        raw = np.stack([g["data"][int(i)] for i in idx]).astype(np.float64)
    keep_m = monotonic_mask(wl_m)
    wl = wl_m[keep_m]
    raw = raw[:, keep_m]
    saturated = (raw >= SATURATION_COUNTS).any(axis=1)
    meas = np.stack([unit_norm(r) for r in raw])

    # ── synthetic (same minerals, all shots), onto the measured axis ──
    wl_s_full = load_wavelength(str(ROOT / args.synthetic_axis))
    keep_s = monotonic_mask(wl_s_full)
    wl_s = wl_s_full[keep_s]
    table = read_sample_table(ROOT / args.synthetic)
    names = table["sample_type_name"].astype(str).to_numpy()
    lo, hi = max(wl.min(), wl_s.min()), min(wl.max(), wl_s.max())
    common = (wl >= lo) & (wl <= hi)
    syn: dict[str, np.ndarray] = {}
    with h5py.File(ROOT / args.synthetic, "r") as f:
        for m in args.minerals:
            rows = np.nonzero(names == m)[0]
            if rows.size == 0:
                raise ValueError(f"mineral {m!r} not in {args.synthetic}")
            S = f["spectra"][rows[0] : rows[-1] + 1][rows - rows[0]].astype(np.float64)
            S = S[:, keep_s]
            on_m = np.stack([np.interp(wl, wl_s, s, left=0.0, right=0.0) for s in S])
            on_m[:, ~common] = 0.0
            syn[m] = np.stack([unit_norm(s) for s in on_m])
    print(
        f"common range {lo:.1f}-{hi:.1f} nm ({int(common.sum())} px); "
        f"saturated shots {int(saturated.sum())}/{len(meas)}"
    )

    # ── PCA + k-means (measured only; synthetic projected) ──
    X = meas[:, common]
    pca = PCA(n_components=args.n_pca, random_state=args.seed).fit(X)
    Z = pca.transform(X)
    km = KMeans(n_clusters=args.k, n_init=20, random_state=args.seed).fit(Z)
    lab = km.labels_
    order = np.argsort(-np.bincount(lab, minlength=args.k))  # cluster 1 = largest
    rank = np.empty(args.k, int)
    rank[order] = np.arange(args.k)
    lab = rank[lab]
    Zs = {m: pca.transform(s[:, common]) for m, s in syn.items()}

    # distance of each synthetic mineral to the nearest measured cluster centre,
    # in units of that cluster's RMS radius (> ~3: outside the measured data)
    centres = np.stack([Z[lab == c].mean(0) for c in range(args.k)])
    radius = np.array(
        [np.sqrt(((Z[lab == c] - centres[c]) ** 2).sum(1).mean()) for c in range(args.k)]
    )
    nearest = {}
    for m, zs in Zs.items():
        d = np.linalg.norm(zs.mean(0)[None] - centres, axis=1) / np.maximum(radius, 1e-12)
        c = int(d.argmin())
        nearest[m] = {"cluster": c + 1, "distance_in_radii": round(float(d[c]), 2)}

    # ── plots ──
    qual = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f",
    ]  # fmt: skip
    syn_col = ["#000000", "#555555", "#a0522d", "#6a3d9a", "#b15928"]
    fig = go.Figure()
    for c in range(args.k):
        m = lab == c
        fig.add_trace(
            go.Scattergl(
                x=wl, y=meas[m].mean(0), mode="lines", line=dict(width=1, color=qual[c % 10]),
                name=f"Mar1A cluster {c + 1} (n={int(m.sum())}, sat {int(saturated[m].sum())})",
                legendgroup="meas", legendgrouptitle_text="measured: k-means class means",
            )
        )  # fmt: skip
    for i, (m, s) in enumerate(syn.items()):
        fig.add_trace(
            go.Scattergl(
                x=wl, y=s.mean(0), mode="lines",
                line=dict(width=1.2, color=syn_col[i % len(syn_col)], dash="dot"),
                name=f"synthetic {m} (mean of {len(s)})",
                legendgroup="syn", legendgrouptitle_text="synthetic (mineral means)",
            )
        )  # fmt: skip
    fig.update_layout(
        title=(
            f"{h5_path.stem}: mean spectra of {args.k} k-means classes ({len(meas)} random shots)"
            " vs synthetic minerals — each spectrum unit-normalised (min-shift, max = 1)"
        ),
        xaxis_title="wavelength [nm]",
        yaxis_title="normalised intensity",
        template="plotly_white",
        height=750,
        legend=dict(groupclick="toggleitem", font=dict(size=11)),
        updatemenus=[
            dict(
                type="buttons", direction="left", x=0.0, y=1.08, xanchor="left",
                buttons=[
                    dict(label="linear y", method="relayout", args=[{"yaxis.type": "linear"}]),
                    dict(label="log y", method="relayout", args=[{"yaxis.type": "log"}]),
                ],
            )
        ],
    )  # fmt: skip
    fig.write_html(out_dir / "spectra.html", include_plotlyjs="cdn")

    ev = pca.explained_variance_ratio_
    fig2 = make_subplots(
        rows=1, cols=2, column_widths=[0.55, 0.45],
        subplot_titles=(
            f"PCA scores (PC1 {100 * ev[0]:.1f} %, PC2 {100 * ev[1]:.1f} %); "
            "synthetic projected",
            "shot positions coloured by cluster",
        ),
    )  # fmt: skip
    for c in range(args.k):
        m = lab == c
        fig2.add_trace(
            go.Scattergl(
                x=Z[m, 0], y=Z[m, 1], mode="markers", marker=dict(size=5, color=qual[c % 10]),
                name=f"cluster {c + 1}", legendgroup=f"c{c}",
            ),
            row=1, col=1,
        )  # fmt: skip
        fig2.add_trace(
            go.Scattergl(
                x=md["X_pos"][idx][m], y=md["Y_pos"][idx][m], mode="markers",
                marker=dict(size=5, color=qual[c % 10]), legendgroup=f"c{c}", showlegend=False,
            ),
            row=1, col=2,
        )  # fmt: skip
    for i, (m, zs) in enumerate(Zs.items()):
        fig2.add_trace(
            go.Scattergl(
                x=zs[:, 0], y=zs[:, 1], mode="markers",
                marker=dict(size=8, symbol="x", color=syn_col[i % len(syn_col)]),
                name=f"synthetic {m}",
            ),
            row=1, col=1,
        )  # fmt: skip
    fig2.update_xaxes(title_text="PC1", row=1, col=1)
    fig2.update_yaxes(title_text="PC2", row=1, col=1)
    fig2.update_xaxes(title_text="X position [mm]", row=1, col=2)
    fig2.update_yaxes(title_text="Y position [mm]", scaleanchor="x2", row=1, col=2)
    fig2.update_layout(template="plotly_white", height=700, title=f"{h5_path.stem}: PCA + k-means")
    fig2.write_html(out_dir / "pca.html", include_plotlyjs="cdn")

    summary = {
        "h5": str(h5_path), "measurement": key, "n": int(len(meas)), "k": args.k,
        "seed": args.seed, "common_range_nm": [float(lo), float(hi)],
        "explained_variance_ratio": [round(float(v), 4) for v in ev],
        "cluster_sizes": np.bincount(lab, minlength=args.k).tolist(),
        "saturated_per_cluster": [int(saturated[lab == c].sum()) for c in range(args.k)],
        "synthetic_nearest_cluster": nearest,
        "shot_index": idx.tolist(), "cluster": (lab + 1).tolist(),
    }  # fmt: skip
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("shot_index", "cluster")}))
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
