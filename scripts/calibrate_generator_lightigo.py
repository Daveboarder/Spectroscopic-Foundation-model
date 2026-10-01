"""
Calibrate the physics_version-2 two-zone generator (`data/two_zone_pipeline.py`)
against a LIGHTIGO HDF5 map of a (near-)pure element sample.  First use:
native bismuth, external_data/Data/matrix_01.h5 (84 shots, 188.6-859.1 nm,
gate delay 0 us, 266 nm laser).

Steps
(i)   Measured reference: the shots with the strongest <element> signal (top
      half of the map by the summed net height of the element's DB lines),
      each baseline-subtracted (rolling minimum), then the median across shots.
      The laser wavelength (+-1.5 nm) and the spectrometer seam are masked.
(ii)  Instrument FWHM: Gaussian(+linear baseline) fits to isolated, unsaturated
      lines of the element -> median per spectrometer channel
      the UV-channel
      median is the fwhm_nm starting point (the fit may rescale it).
(iii) Two-zone fit: theta = (Te1, Ne1, Te2/Te1, l_inner, l_outer, gamma_stark,
      fwhm scale) -> `synthesise_spectrum` on the measured axis (N1 from
      quasi-neutrality, isobaric shell N2 = N1*Te1/Te2, Ne2 from Saha - exactly
      the generator's rules) -> objective
          J = rms_lines[ log10(area_syn / area_meas) - R(lambda) ]   (relative intensities)
            + w_shape * mean_lines[ rms(peak-normalised profile_syn - profile_meas) ]
      where R(lambda) = a + b x + c x^2 (+ d per spectrometer channel), x =
      (lambda - 300)/100, is the unknown spectral response of the uncalibrated
      spectrometer, solved by least squares inside every evaluation (Te is not
      degenerate with it: the Boltzmann lever arm sits between neighbouring
      lines of very different E_k).  The profile term uses each strong line
      normalised to unit peak, so saturated plateaus and wrong widths are
      penalised independently of the intensity scale.
      Random log-uniform search over a wide box, then Nelder-Mead from the best
      starts.  The same fit with l_outer = 0 (one zone) tells whether the cold
      shell is needed for this element.  Parameters that end within 2 % of the
      box are reported as unconstrained.
Composition: the <mineral> row of the sample matrix (same layout as
config/libs_data*.yaml: `Concentrations` + `Uncertainties`), or --composition
when the row does not exist (default: the pure element).

Results: Outputs/calibrate_lightigo_<element>_<ts>/{result.json, overlay_full.png,
overlay_lines.png, line_ratios.png, reference_spectrum.npz} and a printed YAML
snippet with recommended `generation.zones` / `instrument` values.  This script
never edits configs.

Usage:
    uv run python scripts/calibrate_generator_lightigo.py --h5 external_data/Data/matrix_01.h5 --element Bi
    uv run python scripts/calibrate_generator_lightigo.py --h5 ... --element Bi --mineral Bismuth --n_random 600
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from scipy.optimize import curve_fit, minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.atomic_data import mass_to_number_fractions
from data.libs_pipeline import load_sample_types, unit_norm
from data.two_zone_pipeline import _number_density_rows, _saha_equilibrium_ne, synthesise_spectrum

K_B_EV = 8.617333262e-5
T_REF_K = 10000.0  # reference temperature for line-strength ranking
PARAM_NAMES = [
    "log10_Te1",
    "log10_Ne1",
    "te2_ratio",
    "log10_l_inner",
    "log10_l_outer",
    "log10_gamma",
    "fwhm_scale",
]
# widened search box (config ranges are inside it)
BOUNDS = np.array(
    [
        [np.log10(4000.0), np.log10(30000.0)],
        [15.0, 19.5],
        [0.05, 0.95],
        [np.log10(1e-4), np.log10(1.0)],
        [np.log10(1e-7), np.log10(5e-2)],
        [np.log10(1e-3), np.log10(0.3)],
        [0.5, 2.0],
    ]
)


# ─────────────────────────────────────────────────────────────────────────────
# Measured data
# ─────────────────────────────────────────────────────────────────────────────
def load_lightigo(path: str, measurement: str | None = None):
    """(wavelength [nm], shots (n, n_px), metadata dict, measurement key); invalid shots dropped."""
    with h5py.File(path, "r") as f:
        meas = f["measurements"]
        key = measurement or sorted(meas)[0]
        g = meas[key]["libs"]
        wl = g["calibration"][...].astype(np.float64)
        X = g["data"][...].astype(np.float64)
        md = {k: np.asarray(g["metadata"][k][...]) for k in g["metadata"]}
    inv = md.get("Invalid")
    keep = np.ones(len(X), bool) if inv is None else (inv.ravel() == 0)
    return wl, X[keep], md, key


def seam_wavelength(wl: np.ndarray) -> float | None:
    """Wavelength of the first non-increasing step (spectrometer channel seam)."""
    b = np.nonzero(np.diff(wl) <= 0)[0]
    return float(wl[b[0]]) if b.size else None


def subtract_baseline(s: np.ndarray, win_px: int) -> np.ndarray:
    return s - uniform_filter1d(minimum_filter1d(s, win_px), win_px)


def window_area(wl: np.ndarray, s: np.ndarray, lo: float, hi: float, pad: float = 0.6):
    """(area, net peak, noise) of s in [lo, hi]; baseline = 10th percentile of the +-pad nm surroundings, noise = std of their lowest 30 %."""
    m = (wl >= lo) & (wl <= hi)
    bw = (wl >= lo - pad) & (wl <= hi + pad) & ~m
    if m.sum() < 3 or bw.sum() < 6:
        return np.nan, np.nan, np.nan
    base = float(np.percentile(s[bw], 10))
    low = np.sort(s[bw])[: max(4, int(0.3 * bw.sum()))]
    noise = float(np.std(low))
    idx = np.nonzero(m)[0]
    idx = idx[np.argsort(wl[idx])]
    area = float(np.trapezoid(s[idx] - base, wl[idx]))
    return area, float(s[m].max() - base), noise


# ─────────────────────────────────────────────────────────────────────────────
# Line database
# ─────────────────────────────────────────────────────────────────────────────
def db_lines(db_path: str, element: str) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    q = pd.read_sql(
        "select Elem_name, ion_state, Wavelength, Ei, Ek, Ak, gi, gk from QuantParam where Elem_name = ?",
        con,
        params=(element,),
    )
    con.close()
    q["strength"] = q.gk * q.Ak * np.exp(-q.Ek / (K_B_EV * T_REF_K))
    return q


def build_features(
    lines: pd.DataFrame,
    fwhm: float,
    wl_lo: float,
    wl_hi: float,
    laser_nm: float | None,
    seam_nm: float | None,
    cluster_nm: float = 0.25,
) -> list[dict]:
    """Cluster DB lines closer than cluster_nm into one feature; window = lines +- 2 FWHM."""
    L = lines[(lines.Wavelength > wl_lo + 0.5) & (lines.Wavelength < wl_hi - 0.5)].sort_values(
        "Wavelength"
    )
    groups: list[list[pd.Series]] = []
    for _, r in L.iterrows():
        if groups and r.Wavelength - groups[-1][-1].Wavelength <= cluster_nm:
            groups[-1].append(r)
        else:
            groups.append([r])
    feats = []
    for grp in groups:
        wls = [float(r.Wavelength) for r in grp]
        st = [float(r.strength) for r in grp]
        main = grp[int(np.argmax(st))]
        c = float(main.Wavelength)
        if laser_nm is not None and abs(c - laser_nm) < 1.5:
            continue
        if seam_nm is not None and abs(c - seam_nm) < 0.5:
            continue
        feats.append(
            dict(
                centre=c,
                lo=min(wls) - 2 * fwhm,
                hi=max(wls) + 2 * fwhm,
                n_lines=len(grp),
                Ei=float(main.Ei),
                Ek=float(main.Ek),
                stage=str(main.ion_state),
                strength=float(sum(st)),
            )
        )
    return feats


def flag_contaminants(
    feats: list[dict],
    db_path: str,
    contaminants: list[str],
    tol_nm: float = 0.06,
    rel: float = 0.05,
) -> None:
    """Mark features that sit on a strong line (>= rel of that element's strongest) of a contaminant."""
    for f in feats:
        f["blend"] = ""
    for el in contaminants:
        q = db_lines(db_path, el)
        if q.empty:
            continue
        strong = q[q.strength >= rel * q.strength.max()]
        for f in feats:
            near = strong[(strong.Wavelength - f["centre"]).abs() < tol_nm]
            if not near.empty:
                f["blend"] += f"{el} {near.Wavelength.iloc[0]:.3f}; "


# ─────────────────────────────────────────────────────────────────────────────
# Instrument FWHM
# ─────────────────────────────────────────────────────────────────────────────
def _gauss_lin(x, a, mu, sig, b, c):
    return a * np.exp(-0.5 * ((x - mu) / sig) ** 2) + b + c * (x - mu)


def fit_fwhm(wl: np.ndarray, s: np.ndarray, centre: float, half: float = 0.35) -> dict | None:
    m = (wl > centre - half) & (wl < centre + half)
    x, y = wl[m], s[m]
    o = np.argsort(x)
    x, y = x[o], y[o]
    if x.size < 8:
        return None
    k = int(np.argmax(y))
    b0 = float(np.percentile(y, 10))
    p0 = [max(y[k] - b0, 1e-9), float(x[k]), 0.04, b0, 0.0]
    lo = [0.0, x[k] - 0.1, 0.008, -np.inf, -np.inf]
    hi = [np.inf, x[k] + 0.1, 0.3, np.inf, np.inf]
    try:
        popt, _ = curve_fit(_gauss_lin, x, y, p0=p0, bounds=(lo, hi), maxfev=5000)
    except Exception:
        return None
    yhat = _gauss_lin(x, *popt)
    r2 = 1.0 - float(((y - yhat) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12))
    return dict(fwhm=2.3548 * float(popt[2]), mu=float(popt[1]), height=float(popt[0]), r2=r2)


# ─────────────────────────────────────────────────────────────────────────────
# Forward model + objective (module globals so a fork pool can reuse them)
# ─────────────────────────────────────────────────────────────────────────────
_G: dict = {}


def make_row(Te1, Ne1, ratio, l_inner, l_outer, gamma) -> dict:
    els, x = _G["elements"], _G["x_num"]
    N1 = float(
        _number_density_rows(els, x, np.array([Te1]), np.array([Ne1]), _G["db"], _G["nd_max"])[0]
    )
    if l_outer > 0:
        Te2 = Te1 * ratio
        N2 = N1 * Te1 / Te2
        if _G["nd_max"] is not None:
            N2 = min(N2, _G["nd_max"])
        Ne2 = float(_saha_equilibrium_ne(els, x, np.array([Te2]), np.array([N2]), _G["db"])[0])
        return dict(
            plasma_model="two_zone",
            Te1=Te1,
            Ne1=Ne1,
            Te2=Te2,
            Ne2=Ne2,
            l_inner=l_inner,
            l_outer=l_outer,
            N1=N1,
            N2=N2,
            gamma_stark1=gamma,
            gamma_stark2=gamma,
        )
    return dict(
        plasma_model="one_zone",
        Te1=Te1,
        Ne1=Ne1,
        Te2=Te1,
        Ne2=Ne1,
        l_inner=l_inner,
        l_outer=0.0,
        N1=N1,
        N2=N1,
        gamma_stark1=gamma,
        gamma_stark2=gamma,
    )


def synth(theta: np.ndarray, two_zone: bool) -> np.ndarray:
    Te1, Ne1, ratio = 10 ** theta[0], 10 ** theta[1], theta[2]
    l_in, l_out, gamma = 10 ** theta[3], (10 ** theta[4] if two_zone else 0.0), 10 ** theta[5]
    row = make_row(Te1, Ne1, ratio, l_in, l_out, gamma)
    gc = dict(_G["gen_cfg"])
    gc["instrument"] = {**gc["instrument"], "fwhm_nm": _G["fwhm0"] * float(theta[6])}
    return synthesise_spectrum(_G["elements"], _G["mass_fracs"], _G["wl"], row, _G["db"], gc)


def line_areas(wl: np.ndarray, s: np.ndarray, feats: list[dict]) -> np.ndarray:
    return np.array([window_area(wl, s, f["lo"], f["hi"])[0] for f in feats])


def response_design(wl_lines: np.ndarray, channel: np.ndarray, use_channel: bool) -> np.ndarray:
    """Columns of the smooth spectral-response model in log10: 1, x, x^2 (+ channel offset)."""
    x = (np.asarray(wl_lines, float) - 300.0) / 100.0
    cols = [np.ones_like(x), x, x**2]
    if use_channel:
        cols.append(np.asarray(channel, float))
    return np.column_stack(cols)


def fit_response(logr: np.ndarray, ok: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares response coefficients and residuals (nan where not ok)."""
    A = _G["design"][ok]
    coef, *_ = np.linalg.lstsq(A, logr[ok], rcond=None)
    resid = np.full(len(logr), np.nan)
    resid[ok] = logr[ok] - A @ coef
    return coef, resid


def response_at(coef: np.ndarray, wl: np.ndarray, channel: np.ndarray) -> np.ndarray:
    return response_design(wl, channel, _G["use_channel"]) @ coef


def profile_rms(s: np.ndarray) -> float:
    """Mean RMS difference of peak-normalised profiles over the shape lines."""
    acc = []
    for m, ref_n in _G["shape_windows"]:
        w = s[m]
        pk = float(w.max())
        if pk <= 0:
            acc.append(1.0)
            continue
        acc.append(float(np.sqrt(np.mean((w / pk - ref_n) ** 2))))
    return float(np.mean(acc)) if acc else 0.0


def objective_parts(
    theta: np.ndarray, two_zone: bool
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """(J_intensity, J_shape, log ratios, response coefficients, residuals); penalises box violations."""
    lo, hi = BOUNDS[:, 0], BOUNDS[:, 1]
    viol = float(np.sum(np.maximum(0, lo - theta) + np.maximum(0, theta - hi)))
    th = np.clip(theta, lo, hi)
    n = len(_G["feats"])
    nanv = np.full(n, np.nan)
    try:
        s = synth(th, two_zone)
    except Exception:
        return 10.0, 10.0, nanv, np.zeros(_G["design"].shape[1]), nanv
    a_syn = line_areas(_G["wl"], s, _G["feats"])
    a_meas = _G["a_meas"]
    ok = np.isfinite(a_syn) & (a_syn > 0) & (a_meas > 0)
    if ok.sum() < _G["design"].shape[1] + 3:
        return 10.0, 10.0, nanv, np.zeros(_G["design"].shape[1]), nanv
    logr = np.full(n, np.nan)
    logr[ok] = np.log10(a_syn[ok] / a_meas[ok])
    coef, resid = fit_response(logr, ok)
    j_int = float(np.sqrt(np.mean(resid[ok] ** 2)))
    j_shape = profile_rms(s)
    return j_int + 5.0 * viol, j_shape, logr, coef, resid


def objective(theta: np.ndarray, two_zone: bool) -> float:
    j_int, j_shape, *_ = objective_parts(theta, two_zone)
    return j_int + _G["w_shape"] * j_shape


def _eval_random(args):
    theta, two_zone = args
    return objective(theta, two_zone)


# ─────────────────────────────────────────────────────────────────────────────
def parse_composition(spec: str) -> dict[str, float]:
    comp = {}
    for tok in spec.split(","):
        el, val = tok.split(":")
        comp[el.strip()] = float(val)
    tot = sum(comp.values())
    return {k: v / tot for k, v in comp.items()}


def composition_from_matrix(
    xlsx: str, db: str, mineral: str
) -> tuple[dict[str, float], str] | None:
    for st in load_sample_types(xlsx, db):
        if mineral.lower() in st["sample_name"].lower():
            mid = {e: 0.5 * (lo + hi) for e, (lo, hi) in st["concentration_ranges"].items()}
            tot = sum(mid.values())
            return {e: v / tot for e, v in mid.items() if v > 0}, st["sample_name"]
    return None


def _json_ready(o):
    if isinstance(o, dict):
        return {k: _json_ready(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_ready(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def theta_to_params(theta: np.ndarray, two_zone: bool, fwhm0: float) -> dict:
    p = dict(
        Te1_K=10 ** theta[0],
        Ne1_cm3=10 ** theta[1],
        te2_ratio=theta[2],
        Te2_K=10 ** theta[0] * theta[2],
        l_inner_cm=10 ** theta[3],
        l_outer_cm=(10 ** theta[4] if two_zone else 0.0),
        gamma_stark_nm=10 ** theta[5],
        fwhm_nm=fwhm0 * theta[6],
    )
    row = make_row(
        p["Te1_K"],
        p["Ne1_cm3"],
        p["te2_ratio"],
        p["l_inner_cm"],
        p["l_outer_cm"],
        p["gamma_stark_nm"],
    )
    p.update(
        N1_cm3=row["N1"],
        N2_cm3=row["N2"],
        Ne2_cm3=row["Ne2"],
        log10_N1l_inner=np.log10(row["N1"] * p["l_inner_cm"]),
    )
    return p


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--h5", required=True, help="LIGHTIGO HDF5 file")
    ap.add_argument(
        "--efficiency_correction",
        nargs=2,
        metavar=("DEUTERIUM_H5", "HALOGEN_H5"),
        default=None,
        help="apply the lamp-based relative efficiency correction (data/efficiency_correction.py) "
        "to every shot before fitting; the fitted response should then come out flat",
    )
    ap.add_argument(
        "--measurement", default=None, help="measurement key inside the file (default: first)"
    )
    ap.add_argument(
        "--element", default="Bi", help="element whose DB lines define the comparison windows"
    )
    ap.add_argument("--sample_matrix", default="external_data/Source/REE_minerals_oxides.xlsx")
    ap.add_argument(
        "--mineral", default="Bismuth", help="row of the sample matrix (substring match)"
    )
    ap.add_argument(
        "--composition",
        default=None,
        help="fallback mass fractions 'Bi:1.0,S:0.02' when the mineral row is absent (default: pure element)",
    )
    ap.add_argument("--db", default="external_data/Source/LIBS_data.db")
    ap.add_argument(
        "--generator_config",
        default="config/libs_data.yaml",
        help="generation: block for fine_step etc.",
    )
    ap.add_argument(
        "--contaminants",
        default="Na,K,Ca,Mg,Si,Al,Fe,Ti,Mn,Li,O,N,H,Ar",
        help="elements whose strong lines flag a blended feature",
    )
    ap.add_argument(
        "--shot_fraction",
        type=float,
        default=0.5,
        help="fraction of shots (best element signal) kept",
    )
    ap.add_argument("--snr_min", type=float, default=8.0)
    ap.add_argument(
        "--line_rel_min",
        type=float,
        default=1e-3,
        help="DB lines weaker than this fraction of the element's strongest line (at 10 kK) are ignored",
    )
    ap.add_argument(
        "--max_features", type=int, default=60, help="strongest usable line groups kept for the fit"
    )
    ap.add_argument("--w_shape", type=float, default=2.0, help="weight of the profile-shape term")
    ap.add_argument(
        "--n_shape_lines",
        type=int,
        default=10,
        help="strongest unblended lines used for the profile term",
    )
    ap.add_argument(
        "--nd_max",
        type=float,
        default=None,
        help="cap on N per zone (cm^-3); default: generator config value",
    )
    ap.add_argument("--n_random", type=int, default=600)
    ap.add_argument("--n_starts", type=int, default=4)
    ap.add_argument("--n_refine", type=int, default=250, help="Nelder-Mead evaluations per start")
    ap.add_argument("--n_workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="Outputs")
    args = ap.parse_args()

    t0 = time.time()
    root = Path(__file__).resolve().parent.parent
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = root / args.out_dir / f"calibrate_lightigo_{args.element}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    db = str(root / args.db)

    # ---------- composition ----------
    found = composition_from_matrix(str(root / args.sample_matrix), db, args.mineral)
    if found is not None:
        comp, comp_src = found[0], f"{args.sample_matrix} row '{found[1]}'"
    else:
        comp = parse_composition(args.composition or f"{args.element}:1.0")
        comp_src = (
            f"fallback composition (no row matching '{args.mineral}' in {args.sample_matrix})"
        )
    elements = list(comp)
    mass_fracs = np.array([comp[e] for e in elements])
    print(f"Composition: {comp_src}")
    print(
        "   "
        + ", ".join(f"{e} {v:.4f}" for e, v in sorted(comp.items(), key=lambda kv: -kv[1])[:12])
    )

    # ---------- measured ----------
    wl, X, md, key = load_lightigo(str(root / args.h5), args.measurement)
    laser = float(np.ravel(md["Wavelength"])[0]) if "Wavelength" in md else None
    seam = seam_wavelength(wl)
    step = float(np.median(np.diff(wl)[np.diff(wl) > 0]))
    win_px = int(round(8.0 / step)) | 1
    print(
        f"\nMeasured: {args.h5} [{key}] {X.shape[0]} valid shots x {wl.size} px, "
        f"{wl.min():.1f}-{wl.max():.1f} nm, step {step:.4f} nm, seam {seam}, laser {laser} nm, "
        f"gate delay {np.ravel(md.get('Gate Delay us', [np.nan]))[0]} us, energy {np.ravel(md.get('Laser Energy mJ', [np.nan]))[0]} mJ"
    )
    if args.efficiency_correction:
        from data.efficiency_correction import RelativeEfficiencyCorrection

        rec = RelativeEfficiencyCorrection.from_lamp_files(
            root / args.efficiency_correction[0], root / args.efficiency_correction[1]
        )
        X = rec.apply(X, wl, fill="edge")
        print(
            f"Relative efficiency correction applied (valid {rec.valid_range[0]:.1f}-{rec.valid_range[1]:.1f} nm, "
            f"factor range [{rec.factor.min():.3g}, {rec.factor.max():.3g}])"
        )
    net = np.vstack([subtract_baseline(x, win_px) for x in X])

    lines = db_lines(db, args.element)
    n_all = len(lines)
    lines = lines[lines.strength >= args.line_rel_min * lines.strength.max()].reset_index(drop=True)
    print(
        f"{args.element}: {n_all} DB lines, {len(lines)} above {args.line_rel_min:g} of the strongest"
    )
    # provisional features with a guessed FWHM (4 px) to score shots
    fwhm_guess = 4 * step
    feats0 = build_features(lines, fwhm_guess, wl.min(), wl.max(), laser, seam)
    top = sorted(feats0, key=lambda f: -f["strength"])[:8]
    score = np.array([sum(window_area(wl, n, f["lo"], f["hi"])[1] for f in top) for n in net])
    keep = score >= np.quantile(score, 1.0 - args.shot_fraction)
    ref = unit_norm(np.median(net[keep], axis=0))
    print(
        f"Shots kept ({args.element}-rich): {keep.sum()}/{len(X)}; reference = median of kept, unit-normalised"
    )

    # ---------- instrument FWHM ----------
    ch = np.zeros(wl.size, int)
    if seam is not None:
        ch[np.argmax(np.diff(wl) <= 0) + 1 :] = 1
    ref_max = ref.max()
    fits = []
    for f in feats0:
        if f["n_lines"] != 1:
            continue
        others = lines[(lines.Wavelength - f["centre"]).abs().between(0.01, 0.5)]
        if not others.empty:
            continue
        area, peak, noise = window_area(wl, ref, f["lo"], f["hi"])
        if not np.isfinite(peak) or peak / max(noise, 1e-12) < 15 or peak > 0.5 * ref_max:
            continue
        r = fit_fwhm(wl, ref, f["centre"])
        if r and r["r2"] > 0.9 and r["fwhm"] > 1.2 * step:
            r.update(
                centre=f["centre"],
                channel=int(ch[np.argmin(np.abs(wl - f["centre"]))]),
                snr=peak / noise,
            )
            fits.append(r)
    fw = pd.DataFrame(fits)
    if fw.empty:
        print(
            "WARNING: no isolated line passed the isolation test; fitting the 10 strongest single lines instead"
        )
        for f in sorted((f for f in feats0 if f["n_lines"] == 1), key=lambda f: -f["strength"])[
            :10
        ]:
            r = fit_fwhm(wl, ref, f["centre"])
            if r and r["r2"] > 0.9 and r["fwhm"] > 1.2 * step:
                area, peak, noise = window_area(wl, ref, f["lo"], f["hi"])
                r.update(
                    centre=f["centre"],
                    channel=int(ch[np.argmin(np.abs(wl - f["centre"]))]),
                    snr=peak / max(noise, 1e-12),
                )
                fits.append(r)
        fw = pd.DataFrame(fits)
    if fw.empty:
        raise SystemExit("no lines usable for the FWHM fit")
    fwhm_ch = {int(c): float(g.fwhm.median()) for c, g in fw.groupby("channel")}
    fwhm0 = fwhm_ch.get(0, float(fw.fwhm.median()))
    print(
        f"\nInstrument FWHM from {len(fw)} isolated {args.element} lines: "
        + ", ".join(
            f"channel {c}: {v:.4f} nm (n={int((fw.channel == c).sum())})"
            for c, v in fwhm_ch.items()
        )
        + f" -> fwhm0 = {fwhm0:.4f} nm"
    )
    for _, r in fw.sort_values("centre").iterrows():
        print(
            f"   {r.centre:8.3f} nm  FWHM {r.fwhm:.4f}  R2 {r.r2:.3f}  SNR {r.snr:6.1f}  ch {int(r.channel)}"
        )

    # ---------- features for the fit ----------
    feats = build_features(lines, fwhm0, wl.min(), wl.max(), laser, seam)
    flag_contaminants(
        feats,
        db,
        [
            c.strip()
            for c in args.contaminants.split(",")
            if c.strip() and c.strip() != args.element
        ],
    )
    used = []
    for f in feats:
        area, peak, noise = window_area(wl, ref, f["lo"], f["hi"])
        f.update(
            area_meas=area,
            peak_meas=peak,
            snr=(peak / noise if noise and np.isfinite(noise) else np.nan),
        )
        f["used"] = bool(
            np.isfinite(area) and area > 0 and f["snr"] >= args.snr_min and not f["blend"]
        )
        if f["used"]:
            used.append(f)
    if len(used) > args.max_features:
        keep_ids = {id(f) for f in sorted(used, key=lambda f: -f["strength"])[: args.max_features]}
        for f in used:
            f["used"] = id(f) in keep_ids
        used = [f for f in used if f["used"]]
    print(
        f"\nFeatures: {len(feats)} {args.element} line groups in range, {len(used)} used "
        f"(SNR >= {args.snr_min}, unblended); resonance (Ei = 0) among used: {sum(f['Ei'] == 0 for f in used)}"
    )
    for f in sorted(feats, key=lambda f: -f["strength"])[:25]:
        print(
            f"   {f['centre']:8.3f} {f['stage']:>2s} Ei {f['Ei']:.3f} Ek {f['Ek']:.3f} n={f['n_lines']} "
            f"SNR {f['snr']:7.1f} area {f['area_meas']:.4f} {'USED' if f['used'] else 'skip'} {f['blend']}"
        )

    ch_used = np.array([int(ch[np.argmin(np.abs(wl - f["centre"]))]) for f in used])
    use_channel = (ch_used == 1).sum() >= 2 and (ch_used == 0).sum() >= 2
    design = response_design(np.array([f["centre"] for f in used]), ch_used, use_channel)
    shape_windows = []
    for f in sorted(used, key=lambda f: -f["snr"])[: args.n_shape_lines]:
        m = (wl >= f["centre"] - 0.5) & (wl <= f["centre"] + 0.5)
        w = np.clip(ref[m], 0, None)
        shape_windows.append((m, w / max(float(w.max()), 1e-12)))
    print(
        f"Response model: log10 ratio = a + b x + c x^2{' + d [VIS channel]' if use_channel else ''}, "
        f"x = (lambda - 300)/100; profile term on {len(shape_windows)} lines, weight {args.w_shape}"
    )

    gen = yaml.safe_load(open(root / args.generator_config))["generation"]
    gen_cfg = dict(
        fine_step_nm=float(gen.get("fine_step_nm", 0.002)),
        line_window_nm=float(gen.get("line_window_nm", 0.4)),
        adaptive_window=bool(gen.get("adaptive_window", True)),
        min_relative_intensity=float(gen.get("min_relative_intensity", 1e-7)),
        instrument={"profile": "gaussian", "fwhm_nm": fwhm0},
        augment={"noise_sigma": 0.0, "continuum": 0.0},
    )
    _G.update(
        elements=elements,
        mass_fracs=mass_fracs,
        x_num=mass_to_number_fractions(mass_fracs[None, :], elements),
        db=db,
        nd_max=(args.nd_max if args.nd_max is not None else gen.get("number_density_max")),
        wl=wl,
        feats=used,
        a_meas=np.array([f["area_meas"] for f in used]),
        design=design,
        use_channel=use_channel,
        ch_used=ch_used,
        shape_windows=shape_windows,
        gen_cfg=gen_cfg,
        fwhm0=fwhm0,
        w_shape=args.w_shape,
    )

    # ---------- random search ----------
    rng = np.random.default_rng(args.seed)
    thetas = BOUNDS[:, 0] + rng.random((args.n_random, len(BOUNDS))) * (BOUNDS[:, 1] - BOUNDS[:, 0])
    results = {}
    for two_zone in (True, False):
        t1 = time.time()
        jobs = [(th, two_zone) for th in thetas]
        if args.n_workers > 1:
            with mp.get_context("fork").Pool(args.n_workers) as pool:
                J = np.array(pool.map(_eval_random, jobs, chunksize=8))
        else:
            J = np.array([_eval_random(j) for j in jobs])
        order = np.argsort(J)
        print(
            f"\n[{'two-zone' if two_zone else 'one-zone'}] random search: {args.n_random} points in {time.time() - t1:.0f} s; "
            f"best J = {J[order[0]]:.4f}, median {np.median(J):.3f}"
        )
        best = (np.inf, None)
        for k in order[: args.n_starts]:
            res = minimize(
                objective,
                thetas[k],
                args=(two_zone,),
                method="Nelder-Mead",
                options={"maxfev": args.n_refine, "xatol": 1e-3, "fatol": 1e-5},
            )
            print(f"   refine from J={J[k]:.4f} -> {res.fun:.4f} ({res.nfev} evals)")
            if res.fun < best[0]:
                best = (res.fun, np.clip(res.x, BOUNDS[:, 0], BOUNDS[:, 1]))
        j_int, j_shape, logr, coef, resid = objective_parts(best[1], two_zone)
        p = theta_to_params(best[1], two_zone, fwhm0)
        at_bound = [
            PARAM_NAMES[i]
            for i in range(len(BOUNDS))
            if (best[1][i] - BOUNDS[i, 0]) < 0.02 * (BOUNDS[i, 1] - BOUNDS[i, 0])
            or (BOUNDS[i, 1] - best[1][i]) < 0.02 * (BOUNDS[i, 1] - BOUNDS[i, 0])
        ]
        if not two_zone:
            at_bound = [n for n in at_bound if n not in ("te2_ratio", "log10_l_outer")]
        results["two_zone" if two_zone else "one_zone"] = dict(
            J=float(best[0]),
            J_intensity=float(j_int),
            J_shape=float(j_shape),
            theta=best[1],
            params=p,
            logr=logr,
            response_coef=coef,
            resid=resid,
            at_bound=at_bound,
            spectrum=synth(best[1], two_zone),
        )
        print(
            f"   best: J {best[0]:.4f} (intensity rms {j_int:.4f} dex, profile rms {j_shape:.4f}); "
            + ", ".join(f"{k} {v:.4g}" for k, v in p.items())
        )
        print(
            f"   response coefficients (log10): {np.round(coef, 3).tolist()}"
            + (f";  UNCONSTRAINED (at box edge): {at_bound}" if at_bound else "")
        )

    tz, oz = results["two_zone"], results["one_zone"]
    print(
        f"\nShell benefit: J one-zone {oz['J']:.4f} -> two-zone {tz['J']:.4f} "
        f"({100 * (oz['J'] - tz['J']) / max(oz['J'], 1e-9):.0f} % lower)"
    )

    # ---------- figures ----------
    o = np.argsort(wl)
    corr_tz = 10.0 ** response_at(tz["response_coef"], wl, ch)
    corr_oz = 10.0 ** response_at(oz["response_coef"], wl, ch)
    syn_tz = tz["spectrum"] / corr_tz  # synthetic seen through the fitted response
    syn_oz = oz["spectrum"] / corr_oz
    fig, ax = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    for a, (lo_, hi_) in zip(ax, ((wl.min(), 370.0), (370.0, wl.max()))):
        mm = (wl[o] >= lo_) & (wl[o] <= hi_)
        a.plot(
            wl[o][mm],
            np.sqrt(np.clip(ref[o][mm], 0, None)),
            lw=0.6,
            color="k",
            label="measured (median of kept shots, baseline-subtracted)",
        )
        a.plot(
            wl[o][mm],
            np.sqrt(np.clip(syn_tz[o][mm], 0, None)),
            lw=0.6,
            color="C3",
            alpha=0.8,
            label="synthetic two-zone x fitted response",
        )
        for f in used:
            a.axvspan(f["lo"], f["hi"], color="C0", alpha=0.12, lw=0)
        a.set_ylabel("sqrt(intensity)")
        a.set_xlim(lo_, hi_)
    ax[0].legend(loc="upper right", fontsize=8)
    ax[0].set_title(
        f"{args.element}: measured vs synthetic two-zone ({comp_src}); shaded = fitted line windows"
    )
    ax[1].set_xlabel("wavelength (nm)")
    fig.tight_layout()
    fig.savefig(out / "overlay_full.png", dpi=130)
    plt.close(fig)

    top = sorted(used, key=lambda f: -f["strength"])[:16]
    n = len(top)
    fig, axs = plt.subplots(4, 4, figsize=(16, 12))
    for a, f in zip(axs.ravel(), top):
        m = (wl >= f["centre"] - 0.7) & (wl <= f["centre"] + 0.7)
        oo = np.argsort(wl[m])
        a.plot(wl[m][oo], ref[m][oo], "k.-", ms=3, lw=0.8, label="measured")
        a.plot(wl[m][oo], syn_tz[m][oo], "C3-", lw=1.2, label="two-zone")
        a.plot(wl[m][oo], syn_oz[m][oo], "C0--", lw=1.0, label="one-zone")
        a.set_title(
            f"{args.element} {f['stage']} {f['centre']:.3f}  Ei {f['Ei']:.2f} Ek {f['Ek']:.2f} eV",
            fontsize=9,
        )
        a.tick_params(labelsize=7)
    for a in axs.ravel()[n:]:
        a.axis("off")
    axs[0, 0].legend(fontsize=7)
    fig.suptitle(
        "line profiles: measured vs best-fit synthetic (synthetic multiplied by the fitted spectral response)"
    )
    fig.tight_layout()
    fig.savefig(out / "overlay_lines.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
    lam = np.array([f["centre"] for f in used])
    Ek = np.array([f["Ek"] for f in used])
    Ei = np.array([f["Ei"] for f in used])
    res_ = Ei == 0
    a = ax[0]
    a.scatter(
        lam[~res_],
        tz["logr"][~res_],
        c=Ei[~res_],
        cmap="viridis",
        s=40,
        label="excited lower level",
    )
    a.scatter(lam[res_], tz["logr"][res_], marker="s", c="C3", s=50, label="resonance (Ei = 0)")
    lg = np.linspace(wl.min(), wl.max(), 400)
    for c_, ls_, lab_ in ((0, "-", "UV channel"), (1, "--", "VIS channel")):
        if c_ == 1 and not use_channel:
            continue
        a.plot(
            lg,
            response_at(tz["response_coef"], lg, np.full(lg.size, c_)),
            "k",
            ls=ls_,
            lw=1,
            label=f"fitted response {lab_}",
        )
    a.set_xlabel("wavelength (nm)")
    a.set_ylabel("log10(area_syn / area_meas)")
    a.set_title("two-zone: raw line ratios and fitted spectral response")
    a.legend(fontsize=7)
    for a, r, lab in zip(ax[1:], (tz, oz), ("two-zone", "one-zone")):
        d = r["resid"]
        a.scatter(Ek[~res_], d[~res_], c=Ei[~res_], cmap="viridis", s=40)
        a.scatter(Ek[res_], d[res_], marker="s", c="C3", s=50)
        for f, dd in zip(used, d):
            if np.isfinite(dd):
                a.annotate(
                    f"{f['centre']:.1f}",
                    (f["Ek"], dd),
                    fontsize=6,
                    xytext=(2, 2),
                    textcoords="offset points",
                )
        a.axhline(0, color="k", lw=0.5)
        a.set_xlabel("E_k (eV)")
        a.set_ylabel("residual (dex)")
        a.set_title(f"{lab}: residual after response, rms = {r['J_intensity']:.3f} dex")
    fig.tight_layout()
    fig.savefig(out / "line_ratios.png", dpi=130)
    plt.close(fig)

    np.savez(
        out / "reference_spectrum.npz",
        wavelength=wl,
        reference=ref,
        synthetic_two_zone=tz["spectrum"],
        synthetic_one_zone=oz["spectrum"],
        shots_kept=keep,
    )

    # ---------- recommendation ----------
    p = tz["params"]
    rec = dict(
        instrument=dict(profile="gaussian", fwhm_nm=round(float(p["fwhm_nm"]), 4)),
        zones=dict(
            te1=[float(round(0.8 * p["Te1_K"], -2)), float(round(1.2 * p["Te1_K"], -2))],
            ne1=[float(f"{p['Ne1_cm3'] / 3:.2e}"), float(f"{p['Ne1_cm3'] * 3:.2e}")],
            te2_ratio=[
                round(max(0.1, p["te2_ratio"] - 0.1), 2),
                round(min(0.9, p["te2_ratio"] + 0.1), 2),
            ],
            l_inner_cm=[float(f"{p['l_inner_cm'] / 3:.2e}"), float(f"{p['l_inner_cm'] * 3:.2e}")],
            l_outer_cm=[float(f"{p['l_outer_cm'] / 3:.2e}"), float(f"{p['l_outer_cm'] * 3:.2e}")],
            gamma_stark_nm=[
                float(f"{p['gamma_stark_nm'] / 2:.2e}"),
                float(f"{p['gamma_stark_nm'] * 2:.2e}"),
            ],
        ),
    )
    rec = _json_ready(rec)
    print(
        "\nRecommended generation: snippet (point estimate x/÷ 3, Te +-20 %, gamma x/÷ 2; a starting range, not a posterior):"
    )
    print(yaml.safe_dump({"generation": rec}, sort_keys=False, default_flow_style=None).rstrip())
    if tz["at_bound"]:
        print(
            f"WARNING: two-zone parameters at the search-box edge (not constrained by the data): {tz['at_bound']}"
        )

    result = dict(
        h5=args.h5,
        measurement=key,
        element=args.element,
        composition=comp,
        composition_source=comp_src,
        n_shots=int(len(X)),
        n_shots_kept=int(keep.sum()),
        laser_nm=laser,
        seam_nm=seam,
        step_nm=step,
        metadata={k: _json_ready(np.unique(v)[:5]) for k, v in md.items()},
        fwhm_fits=fw.to_dict("records"),
        fwhm_per_channel=fwhm_ch,
        fwhm0=fwhm0,
        features=[{k: v for k, v in f.items()} for f in feats],
        fits={
            k: dict(
                J=v["J"],
                J_intensity=v["J_intensity"],
                J_shape=v["J_shape"],
                theta=dict(zip(PARAM_NAMES, v["theta"])),
                params=v["params"],
                at_bound=v["at_bound"],
                response_coef_log10=v["response_coef"],
                response_columns=["1", "x", "x^2"] + (["vis_channel"] if use_channel else []),
                log_area_ratio={f"{f['centre']:.3f}": lr for f, lr in zip(used, v["logr"])},
                residual={f"{f['centre']:.3f}": rr for f, rr in zip(used, v["resid"])},
            )
            for k, v in results.items()
        },
        recommended=rec,
        bounds=dict(zip(PARAM_NAMES, BOUNDS.tolist())),
        settings=vars(args),
        runtime_s=time.time() - t0,
    )
    (out / "result.json").write_text(json.dumps(_json_ready(result), indent=2))
    print(f"\n[results] {out}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
