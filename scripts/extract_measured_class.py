"""
Extract measured shots of one class (e.g. the epoxy embedding of a polished
section) from a LIBS raster into a small HDF5 file that a data config can add
to the training data (``extra_spectra`` in ``config/libs_data*.yaml``, loaded
by ``data.two_zone_pipeline.load_extra_spectra``).

The shots are taken from a ``scripts/compare_measured_synthetic_pca.py`` run:
every shot of the listed k-means ``--clusters`` in its ``summary.json``.

Output HDF5: ``wavelength`` [n_px] (raw pixel order), ``spectra`` [n, n_px]
(raw counts, float32), ``shot_index``, ``X``, ``Y``, ``X_pos``, ``Y_pos``;
attrs ``label``, ``source_h5``, ``measurement``, ``pca_summary``, ``clusters``.

Usage:
    uv run python scripts/extract_measured_class.py \\
        --h5 .../Data/MAR1A/LIBS/Mar1A.h5 \\
        --summary Outputs/pca_kmeans_Mar1A/summary.json --clusters 4 5 6 7 8 9 \\
        --label epoxid --out external_data/Data/Mar1A_epoxid.h5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--h5", required=True)
    ap.add_argument(
        "--summary", required=True, help="summary.json of compare_measured_synthetic_pca"
    )
    ap.add_argument("--clusters", type=int, nargs="+", required=True, help="1-based cluster ids")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    summary = json.load(open(args.summary))
    shots = np.asarray(summary["shot_index"])
    cluster = np.asarray(summary["cluster"])
    idx = np.sort(shots[np.isin(cluster, args.clusters)])
    with h5py.File(args.h5, "r") as f:
        key = summary.get("measurement") or sorted(f["measurements"])[0]
        g = f["measurements"][key]["libs"]
        wl = g["calibration"][...].astype(np.float64)
        spectra = np.stack([g["data"][int(i)] for i in idx]).astype(np.float32)
        md = {k: g["metadata"][k][...][idx] for k in ("X", "Y", "X_pos", "Y_pos")}

    out = Path(args.out)
    out = out if out.is_absolute() else ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        f.create_dataset("wavelength", data=wl)
        f.create_dataset("spectra", data=spectra, compression="gzip", compression_opts=4)
        f.create_dataset("shot_index", data=idx)
        for k, v in md.items():
            f.create_dataset(k, data=v)
        f.attrs.update(
            label=args.label, source_h5=str(args.h5), measurement=key,
            pca_summary=str(args.summary), clusters=list(args.clusters),
        )  # fmt: skip
    print(
        f"{len(idx)} shots of clusters {args.clusters} -> {out} "
        f"(max counts median {np.median(spectra.max(1)):.0f})"
    )


if __name__ == "__main__":
    main()
