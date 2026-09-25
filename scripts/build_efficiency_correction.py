"""
Build the LIGHTIGO / FireFly relative efficiency correction from the deuterium
and halogen lamp measurements (port of external_data/Data/20240910_REC_FireFly.R,
see data/efficiency_correction.py) and write it out with diagnostic figures.

Outputs (in ``--out_dir``, default Outputs/efficiency_correction_<ts>/):
    efficiency_correction.npz   RelativeEfficiencyCorrection.load() restores it
    efficiency_correction.tsv   wavelength, factor, sensitivity (the R script's table)
    diagnostics.png             deuterium / halogen panels, overlap join, combined
                                factor and relative sensitivity
Optionally ``--apply <lightigo.h5>`` corrects the mean spectrum of a measurement
and adds a before/after panel.

Usage:
    uv run python scripts/build_efficiency_correction.py
    uv run python scripts/build_efficiency_correction.py --apply external_data/Data/matrix_01.h5
    uv run python scripts/build_efficiency_correction.py --overlap interleave   # R-exact join
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.efficiency_correction import (  # noqa: E402
    UV_LAMP_IRRADIANCE,
    UV_LAMP_NM,
    VIS_LAMP_IRRADIANCE,
    VIS_LAMP_NM,
    RelativeEfficiencyCorrection,
    interp_nan,
    load_lightigo_spectrum,
)


def _unorm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.nanmin(x)
    m = np.nanmax(x)
    return x / m if m > 0 else x


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--deuterium", default="external_data/Data/DeuteriumWithCorrection.h5")
    ap.add_argument("--halogen", default="external_data/Data/HalogenWithCorrection.h5")
    ap.add_argument("--uv_range", type=float, nargs=2, default=(180.0, 400.0))
    ap.add_argument("--vis_range", type=float, nargs=2, default=(350.0, 900.0))
    ap.add_argument("--lowess_frac", type=float, default=0.01)
    ap.add_argument("--lowess_iters", type=int, default=3)
    ap.add_argument("--overlap_fit", choices=("linear", "scale"), default="linear")
    ap.add_argument("--overlap", choices=("crossfade", "interleave"), default="crossfade")
    ap.add_argument(
        "--channel_edges",
        type=float,
        nargs="*",
        default=(260.0, 370.0, 460.0, 690.0),
        help="spectrometer gain steps [nm]; smoothing never crosses them (empty list = none)",
    )
    ap.add_argument(
        "--halogen_lowess_frac",
        type=float,
        default=None,
        help="LOWESS span for the halogen spectrum (default: raw, as in the R script)",
    )
    ap.add_argument(
        "--apply",
        default=None,
        help="LIGHTIGO HDF5 whose mean spectrum is corrected for the figure",
    )
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    t0 = time.time()
    root = Path(__file__).resolve().parent.parent
    out = (
        Path(args.out_dir)
        if args.out_dir
        else root / "Outputs" / f"efficiency_correction_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    )
    out.mkdir(parents=True, exist_ok=True)

    rec = RelativeEfficiencyCorrection.from_lamp_files(
        root / args.deuterium,
        root / args.halogen,
        uv_range=tuple(args.uv_range),
        vis_range=tuple(args.vis_range),
        lowess_frac=args.lowess_frac,
        lowess_iters=args.lowess_iters,
        overlap_fit=args.overlap_fit,
        overlap=args.overlap,
        channel_edges_nm=tuple(args.channel_edges) if args.channel_edges else None,
        halogen_lowess_frac=args.halogen_lowess_frac,
    )
    info = rec.info
    print(
        f"Deuterium: {info['deuterium']['file']} (exposure {info['deuterium']['exposure_us']} us, "
        f"median counts UV {info['deuterium_median_counts_uv']:.0f}, overlap {info['deuterium_median_counts_overlap']:.0f})"
    )
    print(
        f"Halogen:   {info['halogen']['file']} (exposure {info['halogen']['exposure_us']} us, "
        f"median counts overlap {info['halogen_median_counts_overlap']:.0f}, VIS {info['halogen_median_counts_vis']:.0f})"
    )
    print(
        f"Overlap {args.vis_range[0]:.0f}-{args.uv_range[1]:.0f} nm: {info['n_overlap_pixels']} pixels, "
        f"vis = {rec.uv_scale[0]:.4g} + {rec.uv_scale[1]:.4g} * uv (rms {info['overlap_fit_rms']:.3g}); join = {args.overlap}"
    )
    print(
        f"Combined correction: {rec.wavelength.size} nodes, valid {rec.valid_range[0]:.2f}-{rec.valid_range[1]:.2f} nm, "
        f"factor range [{rec.factor.min():.3g}, {rec.factor.max():.3g}] "
        f"(sensitivity min/max = {rec.factor.min() / rec.factor.max():.3g})"
    )
    print(
        "Median counts per channel segment: deuterium "
        + ", ".join(f"{v:.0f}" for v in info["deuterium_median_counts_per_segment"])
        + " | halogen "
        + ", ".join(f"{v:.0f}" for v in info["halogen_median_counts_per_segment"])
    )
    if info["deuterium_median_counts_uv"] < 1000:
        print(
            "WARNING: weak deuterium signal in the UV (median < 1000 counts): the UV correction relies on the LOWESS smoothing"
        )

    rec.save(out / "efficiency_correction.npz")
    rec.to_dataframe().to_csv(
        out / "efficiency_correction.tsv", sep="\t", index=False, float_format="%.6g"
    )

    # ---- diagnostics figure ----------------------------------------------------
    wl_d, spec_d, _ = load_lightigo_spectrum(root / args.deuterium)
    wl_h, spec_h, _ = load_lightigo_spectrum(root / args.halogen)
    n_panels = 5 if args.apply else 4
    fig, axs = plt.subplots(n_panels, 1, figsize=(14, 3.6 * n_panels))

    a = axs[0]
    m = (wl_d > args.uv_range[0]) & (wl_d < args.uv_range[1])
    o = np.argsort(wl_d[m])
    a.plot(
        wl_d[m][o],
        _unorm(spec_d[m])[o],
        lw=0.5,
        color="0.6",
        label="deuterium measured (unit-normalised)",
    )
    a.plot(
        rec.uv_nodes[np.argsort(rec.uv_nodes)],
        _unorm(interp_nan(rec.uv_nodes, UV_LAMP_NM, UV_LAMP_IRRADIANCE))[np.argsort(rec.uv_nodes)],
        color="C2",
        label="certified deuterium irradiance (unit-normalised)",
    )
    a.plot(
        rec.uv_nodes[np.argsort(rec.uv_nodes)],
        _unorm(rec.uv_factor)[np.argsort(rec.uv_nodes)],
        color="C3",
        lw=0.8,
        label="UV correction = lamp / smoothed measured (unit-normalised)",
    )
    a.set_title("UV: deuterium lamp")
    a.legend(fontsize=8)

    a = axs[1]
    m = (wl_h > args.vis_range[0]) & (wl_h < args.vis_range[1])
    o = np.argsort(wl_h[m])
    a.plot(
        wl_h[m][o],
        _unorm(spec_h[m])[o],
        lw=0.5,
        color="0.6",
        label="halogen measured (unit-normalised)",
    )
    a.plot(
        rec.vis_nodes[np.argsort(rec.vis_nodes)],
        _unorm(interp_nan(rec.vis_nodes, VIS_LAMP_NM, VIS_LAMP_IRRADIANCE))[
            np.argsort(rec.vis_nodes)
        ],
        color="C2",
        label="certified halogen irradiance (unit-normalised)",
    )
    a.plot(
        rec.vis_nodes[np.argsort(rec.vis_nodes)],
        _unorm(rec.vis_factor)[np.argsort(rec.vis_nodes)],
        color="C3",
        lw=0.8,
        label="VIS correction = lamp / measured (unit-normalised)",
    )
    a.set_title("VIS: halogen lamp")
    a.legend(fontsize=8)

    a = axs[2]
    lo, hi = args.vis_range[0] - 20, args.uv_range[1] + 20
    g = np.linspace(lo, hi, 800)
    a.plot(
        g,
        rec.uv_scale[0] + rec.uv_scale[1] * interp_nan(g, rec.uv_nodes, rec.uv_factor),
        color="C0",
        lw=0.8,
        label="deuterium-based, rescaled",
    )
    a.plot(
        g, interp_nan(g, rec.vis_nodes, rec.vis_factor), color="C1", lw=0.8, label="halogen-based"
    )
    a.plot(g, rec(g), color="k", lw=1.2, label=f"combined ({args.overlap})")
    a.axvspan(args.vis_range[0], args.uv_range[1], color="0.9", zorder=0)
    a.set_title("overlap join")
    a.legend(fontsize=8)

    a = axs[3]
    a.plot(
        rec.wavelength,
        rec.factor / np.nanmax(rec.factor),
        color="k",
        lw=0.8,
        label="correction factor / max",
    )
    a.plot(
        rec.wavelength,
        rec.sensitivity(rec.wavelength),
        color="C3",
        lw=0.8,
        label="relative sensitivity = 1 / factor (max = 1)",
    )
    a.set_yscale("log")
    a.set_title("combined correction and relative sensitivity")
    a.legend(fontsize=8)

    if args.apply:
        wl_s, spec_s, info_s = load_lightigo_spectrum(root / args.apply)
        corr = rec.apply(spec_s, wl_s)
        a = axs[4]
        o = np.argsort(wl_s)
        a.plot(
            wl_s[o],
            _unorm(spec_s)[o],
            lw=0.5,
            color="0.5",
            label="measured mean spectrum (unit-normalised)",
        )
        a.plot(wl_s[o], _unorm(corr)[o], lw=0.5, color="C3", label="corrected (unit-normalised)")
        a.set_title(f"applied to {Path(args.apply).name} [{info_s['measurement']}]")
        a.legend(fontsize=8)
        np.savez(out / "applied_example.npz", wavelength=wl_s, measured=spec_s, corrected=corr)

    for a in axs:
        a.set_xlabel("wavelength (nm)")
    fig.tight_layout()
    fig.savefig(out / "diagnostics.png", dpi=120)
    plt.close(fig)
    print(f"[results] {out}  ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
