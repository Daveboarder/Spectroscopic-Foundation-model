"""
Calibrate the self-absorption knobs of the two-zone generator (`data/two_zone_pipeline.py`)
against TIMA-registered minerals of a real map (first use: Mar1A.h5, 2022 Avantes).

Self-absorption shows in ratios of lines of one element whose thin-limit ratio is fixed by
atomic data: a resonance line (lower level ~0 eV) loses intensity relative to its multiplet
partner as the optical depth grows (Al I 394.40/396.15 thin 0.51 -> 1 when thick). These
ratios do not depend on the composition, and neighbouring lines barely on the spectral
response, so they constrain the optical-depth knobs directly:

    l_inner_cm      inner path length (N1 * l_inner = column density; N1 from quasi-neutrality)
    gamma_stark_nm  Lorentzian HWHM of every non-hydrogen line (peak optical depth ~ 1/width)
    te2_ratio, l_outer_cm   cold isobaric shell (two-zone) or none (one-zone)

Te1 and Ne1 stay at the data config's zone midpoints (hematite fit; Ne agrees with the
H alpha Stark width of the map).

Steps
(i)   Real: shots in grain interiors (TIMA class constant within +-2 cells, peak >= 3000
      counts) of each mineral, rolling-minimum baseline per spectrometer channel, line
      areas in windows centred on the local maximum near each line; per-shot ratios ->
      median and robust spread (1.4826 MAD of log10). Ca I 610.27 is subtracted from the
      Li I 610.36 window via its multiplet partner Ca I 612.22 (thin ratio from the DB),
      because real muscovite carries Ca that the sample matrix does not list.
(ii)  Synthetic: the generator's own transfer (`synthesise_fine_grid`, instrument kernel,
      lamp sensitivity, no jitter / saturation) for the sample-matrix composition of each
      mineral, on short segments of the real pixel axis, same baseline and windows.
(iii) Objective J = mean_w [(log10 r_syn - log10 r_real) / s]^2 with s = max(spread / 2, 0.02)
      dex; log-uniform random search then Nelder-Mead from the best starts, one-zone and
      two-zone. The current config ranges and the proposed ones are compared as
      distributions of synthetic ratios over random draws, like the generator samples them.

Writes Outputs/calibrate_self_absorption_<ts>/{result.json, diagnostics.png} and prints a
YAML snippet for `generation.zones`. Never edits configs.

Usage:
    uv run python scripts/calibrate_self_absorption.py \\
        --h5 /mnt/data/projects/Running_projects/24_0010_Minerals_classification/Data/MAR1A/LIBS/Mar1A.h5 \\
        --registration Outputs/tima_registration_Mar1A \\
        --libs_data_config config/libs_data_minerals_avantes_fw.yaml
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
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from scipy.ndimage import minimum_filter1d, uniform_filter1d  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import data.two_zone_pipeline as tz  # noqa: E402
from data.libs_pipeline import load_sample_types  # noqa: E402
from scripts.eval_tima_transfer import interior_mask  # noqa: E402

# TIMA class index in Outputs/tima_registration_*/tgrid.npy -> sample-matrix row
MINERALS = {0: "Albite", 1: "Quartz", 2: "Muscovite", 3: "Spessartine"}
# name, element, ion, numerator nm, denominator nm, half window nm, minerals, weight
DIAGNOSTICS = [
    ("Al I 394.40/396.15", "Al", "I", 394.40, 396.15, 0.30, ("Albite", "Muscovite", "Spessartine"), 1.0),
    ("Al I 308.22/309.27", "Al", "I", 308.22, 309.27, 0.30, ("Albite", "Muscovite", "Spessartine"), 1.0),
    ("Ca II 396.85/393.37", "Ca", "II", 396.85, 393.37, 0.30, ("Albite", "Spessartine"), 1.0),
    ("Mn I 403.45/403.08", "Mn", "I", 403.45, 403.08, 0.07, ("Spessartine",), 1.0),
    # reported only: Li 610.36/670.78 is set by the Li excitation temperature (real 0.09 is
    # below the 12.5 kK thin limit 0.21), which optical depth cannot lower
    ("Li I 610.36/670.78", "Li", "I", 610.36, 670.78, 0.80, ("Muscovite",), 0.0),
]  # fmt: skip
# line FWHM (half-maximum width on the pixel axis): black cores broaden strong lines
# name, nm, minerals, weight
WIDTHS = [
    ("FWHM Al I 396.15", 396.15, ("Albite", "Muscovite", "Spessartine"), 1.0),
    ("FWHM Al I 394.40", 394.40, ("Albite", "Muscovite", "Spessartine"), 1.0),
    ("FWHM Al I 309.27", 309.27, ("Albite", "Muscovite", "Spessartine"), 1.0),
    ("FWHM Ca II 393.37", 393.37, ("Albite", "Spessartine"), 1.0),
    ("FWHM Si I 288.16", 288.16, ("Quartz", "Albite"), 1.0),
    ("FWHM Li I 670.78", 670.78, ("Muscovite",), 0.5),
]
WIDTH_TOL_DEX = 0.03  # ~7 %: pixel sampling (0.03 nm in UV/VIS, 0.11 nm above 460 nm)
# peak-normalised profiles of the WIDTHS lines, aligned on the peak: catches flat tops and
# self-reversal (a cold shell carves a dip into strong lines; the real lines show none)
SHAPE_OFFSETS = np.linspace(-0.5, 0.5, 101)
SHAPE_TOL = 0.05  # rms of the peak-normalised difference
CA_610 = (610.27, 612.22)  # Ca I contaminant of the Li 610.36 window and its multiplet partner
SEGMENTS = [(285.0, 292.0), (303.0, 314.0), (389.0, 407.0), (605.0, 616.0), (666.0, 675.0)]
BASE_NM = 2.0
_G: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# Spectra -> line areas
# ─────────────────────────────────────────────────────────────────────────────
def channel_bounds(wl: np.ndarray) -> list[tuple[int, int]]:
    cuts = [0, *(np.where(np.diff(wl) < 0)[0] + 1), wl.size]
    return list(zip(cuts[:-1], cuts[1:]))


def debase(x: np.ndarray, wl: np.ndarray) -> np.ndarray:
    """Rolling-minimum + moving-mean baseline (BASE_NM) removed per channel; x [..., n]."""
    x = np.asarray(x, dtype=np.float64).copy()
    for a, b in channel_bounds(wl):
        w = max(5, int(round(BASE_NM / np.median(np.diff(wl[a:b])))))
        x[..., a:b] -= uniform_filter1d(minimum_filter1d(x[..., a:b], w, axis=-1), w, axis=-1)
    return x


def window_area(x: np.ndarray, wl: np.ndarray, nm: float, half: float, search: float) -> np.ndarray:
    """Area within +-half nm of the maximum found within +-search nm of ``nm``; x [..., n]."""
    near = np.where(np.abs(wl - nm) <= search)[0]
    if near.size == 0:
        return np.full(x.shape[:-1], np.nan)
    ref = x.reshape(-1, x.shape[-1]).mean(0)  # one centre for all rows (median-like, robust)
    c = wl[near[np.argmax(ref[near])]]
    m = np.abs(wl - c) <= half
    order = np.argsort(wl[m])
    return np.trapezoid(x[..., m][..., order], wl[m][order], axis=-1)


def thin_ratio(db: sqlite3.Connection, el: str, ion: str, a: float, b: float, T: float) -> float:
    def s(nm: float) -> float:
        rows = db.execute(
            "SELECT Wavelength, Ak, gk, Ek FROM QuantParam WHERE Elem_name=? AND ion_state=? "
            "AND abs(Wavelength - ?) < 0.03", (el, ion, nm),
        ).fetchall()  # fmt: skip
        return sum(A * g * np.exp(-Ek / (8.617333e-5 * T)) / w for w, A, g, Ek in rows)

    return s(a) / s(b)


def ratios(x: np.ndarray, wl: np.ndarray, mineral: str, search: float, ca_k: float) -> dict:
    """Diagnostic ratios of baseline-free spectra x [..., n] (real: search wider for the
    calibration offset; synthetic: lines sit at their DB wavelengths)."""
    out = {}
    for name, el, ion, a, b, half, minerals, _ in DIAGNOSTICS:
        if mineral not in minerals:
            continue
        num = window_area(x, wl, a, half, search if half > 0.1 else 0.06)
        den = window_area(x, wl, b, half, search if half > 0.1 else 0.06)
        if el == "Li" and ca_k > 0:
            num = num - ca_k * window_area(x, wl, CA_610[1], 0.3, search)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[name] = num / den
    return out


def hm_width(x: np.ndarray, wl: np.ndarray, nm: float, search: float, half: float = 0.8) -> float:
    """Full width at half maximum [nm] of the line peaking within +-search of ``nm``
    (linear interpolation of the half-maximum crossings on the pixel axis)."""
    m = np.abs(wl - nm) <= half
    o = np.argsort(wl[m])
    xs, ys = wl[m][o], np.asarray(x, dtype=np.float64)[m][o]
    cand = np.where(np.abs(xs - nm) <= search)[0]
    if cand.size == 0:
        return float("nan")
    i = int(cand[np.argmax(ys[cand])])
    hm = ys[i] / 2.0
    if hm <= 0:
        return float("nan")
    lo = i
    while lo > 0 and ys[lo] > hm:
        lo -= 1
    hi = i
    while hi < ys.size - 1 and ys[hi] > hm:
        hi += 1
    if ys[lo] > hm or ys[hi] > hm:
        return float("nan")
    xl = np.interp(hm, [ys[lo], ys[lo + 1]], [xs[lo], xs[lo + 1]])
    xr = np.interp(hm, [ys[hi], ys[hi - 1]], [xs[hi], xs[hi - 1]])
    return float(xr - xl)


def widths(x: np.ndarray, wl: np.ndarray, mineral: str, search: float) -> dict:
    return {
        name: hm_width(x, wl, nm, search) for name, nm, minerals, _ in WIDTHS if mineral in minerals
    }


def profile(x: np.ndarray, wl: np.ndarray, nm: float, search: float) -> np.ndarray:
    """Peak-normalised line profile on SHAPE_OFFSETS around its peak (parabolic peak
    position, so a calibration offset of the real axis does not matter)."""
    m = np.abs(wl - nm) <= 1.0
    o = np.argsort(wl[m])
    xs, ys = wl[m][o], np.asarray(x, dtype=np.float64)[m][o]
    cand = np.where(np.abs(xs - nm) <= search)[0]
    i = int(cand[np.argmax(ys[cand])])
    c = xs[i]
    if 0 < i < ys.size - 1:
        y0, y1, y2 = ys[i - 1], ys[i], ys[i + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            c = xs[i] + 0.5 * (y0 - y2) / den * 0.5 * (xs[i + 1] - xs[i - 1])
    p = np.interp(c + SHAPE_OFFSETS, xs, ys)
    return p / max(float(p.max()), 1e-30)


def profiles(x: np.ndarray, wl: np.ndarray, mineral: str, search: float) -> dict:
    return {
        name.replace("FWHM", "shape"): profile(x, wl, nm, search)
        for name, nm, minerals, _ in WIDTHS
        if mineral in minerals
    }


# ─────────────────────────────────────────────────────────────────────────────
# Real targets
# ─────────────────────────────────────────────────────────────────────────────
def real_targets(h5: str, registration: str, db: sqlite3.Connection, T: float, slab: int = 1024):
    tg = np.load(Path(registration) / "tgrid.npy")
    with h5py.File(h5, "r") as f:
        g = f["measurements"][sorted(f["measurements"])[0]]["libs"]
        d, md = g["data"], g["metadata"]
        X, Y, inv = (md[k][...].astype(int) for k in ("X", "Y", "Invalid"))
        rows, cols = int(Y.max()) - Y, X
        lab, inter = tg[rows, cols], interior_mask(tg, rows, cols, 2) & (inv == 0)
        ns = (len(X) + slab - 1) // slab
        cnt = np.stack([np.bincount(np.arange(len(X)) // slab, weights=(lab == k) & inter, minlength=ns)
                        for k in MINERALS], 1)  # fmt: skip
        pick = set(np.linspace(0.1 * ns, 0.9 * ns, 10).astype(int))
        for k in MINERALS:
            pick |= set(np.argsort(cnt[:, k])[::-1][: 4 if k < 3 else 10])
        shots = {k: [] for k in MINERALS}
        for s in sorted(pick):
            B = d[s * slab : (s + 1) * slab].astype(np.float32)
            ii = np.arange(s * slab, s * slab + len(B))
            for k in MINERALS:
                m = (lab[ii] == k) & inter[ii] & (B.max(1) >= 3000)
                shots[k] += [B[m]] if m.any() else []
        wl = g["calibration"][...].astype(np.float64)
    ca_k = thin_ratio(db, "Ca", "I", CA_610[0], CA_610[1], T)
    out = {}
    for k, name in MINERALS.items():
        S = np.concatenate(shots[k])
        base = debase(np.median(S, 0), wl)
        per_shot = {}
        for c in range(0, len(S), 2048):  # baseline per shot in chunks (memory)
            r = ratios(debase(S[c : c + 2048], wl), wl, name, 0.2, ca_k)
            for key, v in r.items():
                per_shot.setdefault(key, []).append(v)
        med = ratios(base, wl, name, 0.2, ca_k)
        out[name] = {"n_shots": int(len(S)), "spectrum": base, "diag": {}}
        for key, v in widths(base, wl, name, 0.15).items():
            out[name]["diag"][key] = {"ratio": v, "log_spread": 2.0 * WIDTH_TOL_DEX}
        out[name]["profiles"] = profiles(base, wl, name, 0.15)
        for key, v in med.items():
            v_shot = np.concatenate(per_shot[key])
            lv = np.log10(v_shot[np.isfinite(v_shot) & (v_shot > 0)])
            spread = float(1.4826 * np.median(np.abs(lv - np.median(lv)))) if lv.size else np.nan
            out[name]["diag"][key] = {"ratio": float(v), "log_spread": spread}
    return out, wl, ca_k


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic forward model
# ─────────────────────────────────────────────────────────────────────────────
def composition(sample_matrix: str, db_path: str, name: str) -> tuple[list[str], np.ndarray]:
    st = {s["sample_name"]: s for s in load_sample_types(sample_matrix, db_path)}
    cr = st[name]["concentration_ranges"]
    els = [e for e in cr if sum(cr[e]) > 0]
    w = np.array([0.5 * (cr[e][0] + cr[e][1]) for e in els])
    return els, w / w.sum()


def make_row(els, x, Te1, Ne1, ratio, l_inner, l_outer, gamma, db) -> dict:
    """Zone row with the generator's rules (quasi-neutral N1, isobaric shell, Saha Ne2)."""
    N1 = float(tz._number_density_rows(els, x[None], np.array([Te1]), np.array([Ne1]), db, None)[0])
    if l_outer <= 0:
        return dict(plasma_model="one_zone", Te1=Te1, Ne1=Ne1, Te2=Te1, Ne2=Ne1, l_inner=l_inner,
                    l_outer=0.0, N1=N1, N2=N1, gamma_stark1=gamma, gamma_stark2=gamma)  # fmt: skip
    Te2 = Te1 * ratio
    N2 = N1 * Te1 / Te2
    Ne2 = float(tz._saha_equilibrium_ne(els, x[None], np.array([Te2]), np.array([N2]), db)[0])
    return dict(plasma_model="two_zone", Te1=Te1, Ne1=Ne1, Te2=Te2, Ne2=Ne2, l_inner=l_inner,
                l_outer=l_outer, N1=N1, N2=N2, gamma_stark1=gamma, gamma_stark2=gamma)  # fmt: skip


def synth_spectrum(mineral: str, row: dict) -> tuple[np.ndarray, np.ndarray]:
    """Radiance x lamp sensitivity on the real pixels of SEGMENTS (concatenated, baseline-free)."""
    els, x = _G["comp"][mineral]
    inner, outer = tz.zone_states_from_row(row)
    parts, wls = [], []
    for wl, sens in zip(_G["seg_wl"], _G["sens_list"]):
        grid = tz.make_fine_grid(wl, _G["fine_step"])
        rad = tz.synthesise_fine_grid(els, x, grid, inner, outer, _G["db"], line_window_nm=_G["line_window"],
                                      adaptive_window=True, stark=_G["stark"])  # fmt: skip
        rad = tz.broaden_instrument(rad, grid, _G["fine_step"], _G["instrument"])
        s = np.interp(wl, grid, rad) * sens
        parts.append(debase(s, wl))
        wls.append(wl)
    return np.concatenate(wls), np.concatenate(parts)


def synth_ratios(theta: np.ndarray, two_zone: bool) -> dict:
    l_in, gamma = 10 ** theta[0], 10 ** theta[1]
    ratio, l_out = (float(theta[2]), 10 ** theta[3]) if two_zone else (1.0, 0.0)
    out = {}
    for m in _G["fit_minerals"]:
        els, x = _G["comp"][m]
        row = make_row(els, x, _G["Te1"], _G["Ne1"], ratio, l_in, l_out, gamma, _G["db"])
        wl, s = synth_spectrum(m, row)
        out[m] = {k: float(v) for k, v in ratios(s, wl, m, 0.06, 0.0).items()}
        out[m].update(widths(s, wl, m, 0.06))
        for key, p in profiles(s, wl, m, 0.06).items():  # rms vs the real profile
            out[m][key] = float(np.sqrt(np.mean((p - _G["real_profiles"][m][key]) ** 2)))
    return out


def objective(theta: np.ndarray, two_zone: bool) -> float:
    lo, hi = _G["box"][: len(theta)].T
    if np.any(theta < lo) or np.any(theta > hi):
        return 1e6
    try:
        r = synth_ratios(theta, two_zone)
    except Exception:  # noqa: BLE001 - a failed synthesis is a bad point, not a crash
        return 1e6
    terms, weights = [], []
    keys = [(d[0], d[-1]) for d in DIAGNOSTICS] + [(d[0], d[-1]) for d in WIDTHS]
    for name, w in keys:
        if w <= 0:
            continue
        for m, t in _G["targets"].items():
            if name in t and name in r.get(m, {}) and r[m][name] > 0 and t[name]["ratio"] > 0:
                s = max(t[name]["log_spread"] / 2.0, 0.02)
                terms.append(((np.log10(r[m][name]) - np.log10(t[name]["ratio"])) / s) ** 2)
                weights.append(w)
    for name, _, minerals, w in WIDTHS:
        key = name.replace("FWHM", "shape")
        for m in _G["targets"]:
            if m in minerals and key in r.get(m, {}):
                terms.append((r[m][key] / SHAPE_TOL) ** 2)
                weights.append(w)
    return float(np.average(terms, weights=weights)) if terms else 1e6


def _init(g: dict) -> None:
    _G.update(g)


def _eval(args):
    theta, two_zone = args
    return objective(np.asarray(theta), two_zone)


def _refine(args):
    theta0, two_zone, n_eval = args
    res = minimize(objective, np.asarray(theta0), args=(two_zone,), method="Nelder-Mead",
                   options={"maxfev": n_eval, "xatol": 1e-3, "fatol": 1e-4})  # fmt: skip
    return res.x.tolist(), float(res.fun)


def draws_from_zones(zones: dict, n: int, rng: np.random.Generator, two_zone: bool) -> np.ndarray:
    """theta rows sampled like the generator samples a config's zone ranges."""
    lu = lambda a, b: rng.uniform(np.log10(a), np.log10(b), n)  # noqa: E731
    th = [lu(*zones["l_inner_cm"]), lu(*zones["gamma_stark_nm"])]
    if two_zone:
        th += [rng.uniform(*zones["te2_ratio"], n), lu(*zones["l_outer_cm"])]
    return np.stack(th, 1)


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--h5", required=True)
    ap.add_argument(
        "--registration", required=True, help="dir with tgrid.npy (eval_tima_transfer register)"
    )
    ap.add_argument("--libs_data_config", required=True)
    ap.add_argument("--n_random", type=int, default=400)
    ap.add_argument("--n_starts", type=int, default=3)
    ap.add_argument("--n_refine", type=int, default=150)
    ap.add_argument(
        "--n_check", type=int, default=120, help="random draws per range set in the check"
    )
    ap.add_argument("--n_workers", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="Outputs")
    a = ap.parse_args()

    t0 = time.time()
    cfg = yaml.safe_load(open(ROOT / a.libs_data_config))
    gen, paths = cfg["generation"], cfg["paths"]
    db_path = str(ROOT / paths["db"])
    db = sqlite3.connect(db_path)
    zones = gen["zones"]
    Te1 = float(np.mean(zones["te1"]))
    Ne1 = float(10 ** np.mean(np.log10(zones["ne1"])))
    print(f"Te1 {Te1:.0f} K, Ne1 {Ne1:.2e} cm^-3 (config midpoints); real targets from {a.h5}")

    real, wl, ca_k = real_targets(a.h5, a.registration, db, Te1)
    targets = {m: v["diag"] for m, v in real.items() if v["diag"]}
    for name, *_ in WIDTHS:
        vals = ", ".join(f"{m} {t[name]['ratio']:.3f} nm" for m, t in targets.items() if name in t)
        print(f"  {name:22s} real {vals}")
    for name, el, ion, x1, x2, *_ in DIAGNOSTICS:
        thin = thin_ratio(db, el, ion, x1, x2, Te1)
        vals = ", ".join(f"{m} {t[name]['ratio']:.3f} (+-{t[name]['log_spread']:.2f} dex)"
                         for m, t in targets.items() if name in t)  # fmt: skip
        print(f"  {name:22s} thin {thin:.3f} | real {vals}")

    # lamp sensitivity on the whole axis (it is normalised to its maximum), then cut per segment
    resp = (gen.get("detector") or {}).get("response")
    sens_full = tz.detector_sensitivity(wl, resp) if resp else np.ones_like(wl)
    seg_wl, sens_list = [], []
    for lo, hi in SEGMENTS:
        m = np.where((wl >= lo) & (wl <= hi))[0]
        o = np.argsort(wl[m])
        seg_wl.append(wl[m][o])
        sens_list.append(sens_full[m][o])
    fit_minerals = list(targets)
    comp = {}
    for m in fit_minerals:
        els, w = composition(str(ROOT / paths["sample_matrix"]), db_path, m)
        comp[m] = (els, tz.mass_to_number_fractions(w, els))
    box = np.array([[-3.0, 0.0], [-3.0, -0.5], [0.15, 0.95], [-6.0, -2.0]])
    g = dict(comp=comp, targets={m: targets[m] for m in fit_minerals}, fit_minerals=fit_minerals,
             seg_wl=seg_wl, sens_list=sens_list, db=db_path, Te1=Te1, Ne1=Ne1, box=box,
             fine_step=float(gen.get("fine_step_nm", 0.002)), line_window=float(gen.get("line_window_nm", 0.4)),
             instrument=gen.get("instrument", tz.DEFAULT_INSTRUMENT), stark=gen.get("stark"),
             real_profiles={m: real[m]["profiles"] for m in targets})  # fmt: skip
    _init(g)

    rng = np.random.default_rng(a.seed)
    results = {}
    with mp.Pool(a.n_workers, initializer=_init, initargs=(g,)) as pool:
        for two_zone in (False, True):
            k = 4 if two_zone else 2
            th = rng.uniform(box[:k, 0], box[:k, 1], (a.n_random, k))
            J = np.array(pool.map(_eval, [(t, two_zone) for t in th], chunksize=4))
            starts = th[np.argsort(J)[: a.n_starts]]
            refined = pool.map(_refine, [(s.tolist(), two_zone, a.n_refine) for s in starts])
            best = min(refined, key=lambda r: r[1])
            tag = "two_zone" if two_zone else "one_zone"
            results[tag] = {"theta": best[0], "J": best[1], "J_random_best": float(J.min()),
                            "ratios": synth_ratios(np.asarray(best[0]), two_zone),
                            "random": {"theta": th.tolist(), "J": J.tolist()}}  # fmt: skip
            t = best[0]
            desc = f"l_inner {10 ** t[0]:.3g} cm, gamma {10 ** t[1]:.3g} nm"
            if two_zone:
                desc += f", te2_ratio {t[2]:.2f}, l_outer {10 ** t[3]:.3g} cm"
            print(
                f"[{tag}] J {best[1]:.3f} (random best {J.min():.3f}): {desc}  ({time.time() - t0:.0f} s)"
            )

        # distributions of the current ranges vs proposed ranges around the best fit
        win = (
            "two_zone" if results["two_zone"]["J"] < 0.8 * results["one_zone"]["J"] else "one_zone"
        )
        bt = results[win]["theta"]
        two = win == "two_zone"
        prop = {"l_inner_cm": [10 ** (bt[0] - 0.3), 10 ** (bt[0] + 0.3)],
                "gamma_stark_nm": [10 ** (bt[1] - 0.2), 10 ** (bt[1] + 0.2)]}  # fmt: skip
        if two:
            prop["te2_ratio"] = [max(0.15, bt[2] - 0.1), min(0.95, bt[2] + 0.1)]
            prop["l_outer_cm"] = [10 ** (bt[3] - 0.3), 10 ** (bt[3] + 0.3)]
        cur_two = gen.get("plasma_model", "mixed") != "one_zone"
        check = {}
        for tag, zz, tw in (("current", zones, cur_two), ("proposed", prop, two)):
            th = draws_from_zones(zz, a.n_check, rng, tw)
            rr = pool.starmap(synth_ratios, [(t, tw) for t in th], chunksize=2)
            check[tag] = {m: {name: np.nanpercentile([r[m][name] for r in rr], [10, 50, 90]).tolist()
                              for name in rr[0][m]} for m in fit_minerals}  # fmt: skip

    out = Path(a.out_dir) / f"calibrate_self_absorption_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    print("\nRatio distributions (10/50/90 %) vs real median:")
    for m in fit_minerals:
        for name in targets[m]:
            c, p = check["current"][m][name], check["proposed"][m][name]
            print(
                f"  {m:11s} {name:22s} real {targets[m][name]['ratio']:.3f} | current "
                f"{c[0]:.3f}/{c[1]:.3f}/{c[2]:.3f} | proposed {p[0]:.3f}/{p[1]:.3f}/{p[2]:.3f}"
            )

    print("Profile rms vs real (peak-normalised, median over draws; tolerance %.2f):" % SHAPE_TOL)
    for m in fit_minerals:
        for key in [k for k in check["current"][m] if k.startswith("shape")]:
            print(f"  {m:11s} {key:22s} current {check['current'][m][key][1]:.3f} | "
                  f"proposed {check['proposed'][m][key][1]:.3f}")  # fmt: skip

    # figure: diagnostic windows, real median vs best synthetic (each window peak-normalised)
    fig, axes = plt.subplots(
        len(fit_minerals), len(SEGMENTS), figsize=(18, 3.0 * len(fit_minerals))
    )
    for i, m in enumerate(fit_minerals):
        els, x = comp[m]
        row = make_row(els, x, Te1, Ne1, bt[2] if two else 1.0, 10 ** bt[0], 10 ** bt[3] if two else 0.0,
                       10 ** bt[1], db_path)  # fmt: skip
        swl, s = synth_spectrum(m, row)
        for j, (lo, hi) in enumerate(SEGMENTS):
            ax = axes[i, j]
            mr, ms = (wl >= lo) & (wl <= hi), (swl >= lo) & (swl <= hi)
            yr, ys = real[m]["spectrum"][mr], s[ms]
            o = np.argsort(wl[mr])
            ax.plot(
                wl[mr][o], yr[o] / max(yr.max(), 1e-12), color="#52514e", lw=1, label="real median"
            )
            ax.plot(
                swl[ms],
                ys / max(ys.max(), 1e-12),
                color="#2a78d6",
                lw=1,
                label=f"synthetic ({win})",
            )
            ax.set_xlim(lo, hi)
            ax.set_title(f"{m} {lo:.0f}-{hi:.0f} nm", fontsize=9)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "diagnostics.png", dpi=110)

    json.dump({"config": a.libs_data_config, "Te1": Te1, "Ne1": Ne1, "ca_610_over_612_thin": ca_k,
               "targets": targets, "n_shots": {m: v["n_shots"] for m, v in real.items()},
               "fits": {k: {kk: vv for kk, vv in v.items() if kk != "random"} for k, v in results.items()},
               "winner": win, "proposed_zones": prop, "check": check,
               "random_search": {k: v["random"] for k, v in results.items()}},
              open(out / "result.json", "w"), indent=1, default=float)  # fmt: skip
    print(f"\nRecommended generation.zones ({win}; {out}):")
    print(
        yaml.safe_dump(
            {k: [float(f"{v:.3g}") for v in vv] for k, vv in prop.items()}, sort_keys=False
        )
    )
    if not two:
        print("plasma_model: one_zone   # the cold shell does not improve the fit")


if __name__ == "__main__":
    main()
