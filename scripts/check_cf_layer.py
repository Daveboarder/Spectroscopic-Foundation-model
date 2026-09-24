"""
Sanity checks for the calibration-free solver package (``cf/``).

  (i)   build ``CFTables`` for the 38 measured-cache elements
        (``external_data/cache/measured_cache_942cec6d2499.h5``; fixed list
        as fallback) and verify U(T) against ``partition_function_cached``
  (ii)  synthetic round-trip on exact thin areas from
        ``data.plasma_physics.thin_line_intensities`` (Fe 0.7 / Cr 0.18 /
        Ni 0.09 / C 0.03 mass fractions, T = 11000 K, Ne = 2e17, l = 0.1 cm):
        T within 1 %, log10 Ne within 0.05, majors within 5 %; then with
        self-absorbed areas and ``sa_correction=True``: majors within 10 %
  (iii) torch layer == numpy solver (< 1e-6 relative) on random weights,
        batch of 3 measured token spectra (falls back to synthetic tokens)
  (iv)  finite gradients w.r.t. weights, T0, log10_Ne0, log10_Nl0 and a
        central-difference Jacobian check on a tiny float64 problem
  (v)   degenerate cases: element with a single line, element with only
        stage-II lines, all weights zero — no NaNs, source 'seed'
  (vi)  curve-of-growth table vs. ``plasma_physics.curve_of_growth_factor``

Exit code 1 on any failure.

Usage:
    uv run python scripts/check_cf_layer.py
    uv run python scripts/check_cf_layer.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cf.layer import SahaBoltzmannLayer                              # noqa: E402
from cf.solver_np import curve_of_growth_table, saha_boltzmann_solve_np  # noqa: E402
from cf.tables import build_cf_tables                                # noqa: E402
from data.atomic_data import atomic_mass, mass_to_number_fractions   # noqa: E402
from data.libs_pipeline import _META_COLS, partition_function_cached  # noqa: E402
from data.line_tokenization import N_FEATURES, atomic_number         # noqa: E402
from data.plasma_physics import (                                    # noqa: E402
    NM_TO_CM, curve_of_growth_factor, doppler_sigma_nm, line_set_for_element,
    number_density_from_ne, thin_line_intensities, voigt_peak,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "external_data" / "Source" / "LIBS_data.db"
DEFAULT_MEASURED = ROOT / "external_data" / "cache" / "measured_cache_942cec6d2499.h5"
DEFAULT_TOKENS = ROOT / "external_data" / "cache" / "line_tokens_2a03a3539d56.h5"
FALLBACK_ELEMENTS = [
    "Fe", "C", "Mn", "Si", "P", "S", "Ni", "Cr", "Cu", "Mo", "V", "Ti", "Al", "W", "As", "Sn",
    "Co", "Pb", "B", "Ta", "Ca", "Mg", "Zn", "Ce", "La", "N", "O", "Ag", "Au", "Ba", "Cd", "Hg",
    "Nd", "Pr", "Sm", "Sr", "Y", "Pt",
]

_failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    if not cond:
        _failures.append(msg)


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────
def element_names_from_cache(path: Path) -> list[str]:
    try:
        with h5py.File(path, "r") as f:
            cols = json.loads(f["sample_table"].attrs["columns"])
        names = [c for c in cols if c not in _META_COLS]
        print(f"  element names from {path.name}: {len(names)}")
        return names
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN: cannot read {path} ({exc}); using the fixed 38-element list")
        return list(FALLBACK_ELEMENTS)


def synthetic_tokens(
    names: list[str], comp: dict[str, float], T: float, Ne: float, l: float, db: str,
    n_per_element: int = 40, area_scale: float = 3.7e5, gamma_nm: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Fake token tensor from exact thin-plasma line intensities.

    Returns (tokens [L,14], tau0_true [L], sigma_true [L], N*l)."""
    wmass = np.zeros(len(names))
    for k, v in comp.items():
        wmass[names.index(k)] = v
    x = mass_to_number_fractions(wmass, names)
    lsets = {e: line_set_for_element(e, T, Ne, db) for e in comp}
    r_II = np.array([lsets[e].r_II if e in lsets else 0.0 for e in names])
    N = number_density_from_ne(Ne, x, r_II)
    rows, tau, sig = [], [], []
    for e in comp:
        d = thin_line_intensities(e, T, Ne, x[names.index(e)], N, l, db)
        idx = np.argsort(d["I_int"])[::-1][:n_per_element]
        for i in idx:
            row = np.zeros(N_FEATURES)
            row[0] = d["wl"][i]; row[1] = d["Ei"][i]; row[2] = d["Ek"][i]
            row[3] = np.log10(d["gi"][i]); row[4] = np.log10(d["gk"][i]); row[5] = np.log10(d["Ak"][i])
            row[6] = np.log10(d["I_int"][i]); row[7] = atomic_number(e)
            row[8] = 0.0 if d["is_I"][i] else 1.0
            row[9] = d["I_int"][i] * area_scale
            row[10] = 0.05; row[11] = 0.99
            rows.append(row)
            s = doppler_sigma_nm(d["wl"][i], T, atomic_mass(e))
            sig.append(s)
            tau.append(d["kappa_int"][i] * voigt_peak(s, gamma_nm) / NM_TO_CM * l)
    return np.asarray(rows), np.asarray(tau), np.asarray(sig), N * l


# ─────────────────────────────────────────────────────────────────────────────
# checks
# ─────────────────────────────────────────────────────────────────────────────
def check_tables(names: list[str], db: str):
    print("\n(i) CFTables")
    t = time.time()
    tab = build_cf_tables(names, db)
    print(f"  built in {time.time() - t:.2f} s: E={tab.n_elements}, U{tab.U.shape}, "
          f"cog table {tab.cog_phi_hat.shape}")
    check(tab.z_to_elem.shape == (120,) and (tab.z_to_elem >= 0).sum() == len(names), "z_to_elem covers the targets")
    check(np.allclose(tab.T_grid[[0, -1]], [3000.0, 30000.0]) and np.isclose(tab.T_step, 100.0), "T grid 3000..30000 step 100")
    err = 0.0
    for e in names[:10]:
        i = names.index(e)
        for T in (4000.0, 11000.0, 25000.0):
            U_ref = partition_function_cached(e, T, db)
            U_tab = tab.partition_functions(T)[i]
            err = max(err, float(np.max(np.abs(U_tab / np.array(U_ref) - 1.0))))
    check(err < 1e-12, f"U(T) grid == partition_function_cached on grid points (max rel {err:.1e})")
    T = 11050.0
    U_ref = np.array(partition_function_cached("Fe", T, db))
    U_tab = tab.partition_functions(T)[names.index("Fe")]
    err = float(np.max(np.abs(U_tab / U_ref - 1.0)))
    check(err < 1e-3, f"U(T) linear interpolation between grid points (Fe, 11050 K: rel {err:.1e})")
    check(np.all(tab.lod > 0) and tab.lod[names.index("Fe")] == 1e-3, "LOD vector from config/element_lod.yaml")
    check(abs(tab.mass_amu[names.index("Fe")] - 55.845) < 1e-6, "atomic masses")
    return tab


def check_cog(tab):
    print("\n(vi) curve-of-growth table")
    tau = np.array([0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0])
    worst = 0.0
    for a in (0.13, 0.77, 3.3, 7.9, 25.5):
        sigma, gamma = 1e-3, a * 1e-3
        ref = curve_of_growth_factor(tau, sigma, gamma, half_width_factor=2000.0, n_grid=2_000_001)
        f = curve_of_growth_table(tau, np.full_like(tau, a), tab)
        worst = max(worst, float(np.max(np.abs(f / ref - 1.0))))
    check(worst < 1e-4, f"table f(tau0) vs plasma_physics.curve_of_growth_factor (±2000 FWHM, 2e6 pts): max rel {worst:.1e}")
    f_small = curve_of_growth_table(np.array([1e-6, 0.0]), np.array([5.0, 5.0]), tab)
    check(np.allclose(f_small, 1.0, atol=1e-5), "f(tau0 -> 0) = 1")


def check_round_trip(names, tab, db):
    print("\n(ii) synthetic round-trip")
    comp = {"Fe": 0.7, "Cr": 0.18, "Ni": 0.09, "C": 0.03}
    T_true, Ne_true, l = 11000.0, 2e17, 0.1
    tok, tau_true, sig_true, Nl = synthetic_tokens(names, comp, T_true, Ne_true, l, db)
    L = tok.shape[0]
    print(f"  {L} lines, true tau0 in [{tau_true.min():.2g}, {tau_true.max():.3g}], "
          f"{(tau_true > 1).sum()} lines with tau0 > 1, log10(N l) = {np.log10(Nl):.3f}")
    fv = np.ones(L)
    w = np.ones(L)
    idx = {e: names.index(e) for e in comp}

    # thin areas, no correction
    r = saha_boltzmann_solve_np(tok, fv, w, tab, C0=None, T0=10000.0, log10_Ne0=17.0,
                                log10_Nl0=np.log10(Nl), sa_correction=False)
    print(f"  thin: T={r.T:.1f} K (true {T_true}), log10 Ne={r.log10_Ne:.3f} (true {np.log10(Ne_true):.3f}); "
          + ", ".join(f"{e}={r.mass_fractions[i]:.4f}/{comp[e]}" for e, i in idx.items()))
    check(abs(r.T / T_true - 1) < 0.01, f"thin: T within 1 % ({abs(r.T / T_true - 1):.2%})")
    check(abs(r.log10_Ne - np.log10(Ne_true)) < 0.05, f"thin: log10 Ne within 0.05 ({abs(r.log10_Ne - np.log10(Ne_true)):.3f})")
    for e, i in idx.items():
        rel = abs(r.mass_fractions[i] / comp[e] - 1)
        check(rel < 0.05, f"thin: {e} within 5 % ({rel:.2%})")
    check(np.all(r.n_lines_used[list(idx.values())] == 40) and np.all(r.source[list(idx.values())] == "cf"), "thin: 40 lines used per element, source 'cf'")
    check(np.all(r.mass_fractions[[i for i in range(len(names)) if i not in idx.values()]] == 0), "thin: elements without lines -> 0 (no C0)")
    tau_rel = float(np.max(np.abs(r.tau0 / tau_true - 1)))
    check(tau_rel < 0.05, f"thin: tau0 at the recovered state within 5 % of the true tau0 (max {tau_rel:.2%})")

    # priors off -> exact
    r0 = saha_boltzmann_solve_np(tok, fv, w, tab, C0=None, T0=10000.0, log10_Ne0=17.0,
                                 log10_Nl0=np.log10(Nl), sa_correction=False, prior_T=0.0, prior_Ne=0.0)
    check(abs(r0.T / T_true - 1) < 1e-4 and abs(r0.log10_Ne - np.log10(Ne_true)) < 1e-4,
          f"thin, priors off: exact recovery (T {r0.T:.2f}, log10 Ne {r0.log10_Ne:.5f})")

    # self-absorbed areas
    f_true = curve_of_growth_factor(tau_true, sig_true, 0.01, half_width_factor=2000.0, n_grid=2_000_001)
    tok_sa = tok.copy()
    tok_sa[:, 9] *= f_true
    wmass = np.zeros(len(names))
    for e, i in idx.items():
        wmass[i] = comp[e]
    r_off = saha_boltzmann_solve_np(tok_sa, fv, w, tab, C0=None, T0=10000.0, log10_Ne0=17.0,
                                    log10_Nl0=np.log10(Nl), sa_correction=False)
    print(f"  self-absorbed, no correction: T={r_off.T:.1f}, log10 Ne={r_off.log10_Ne:.3f}; "
          + ", ".join(f"{e}={r_off.mass_fractions[i]:.4f}" for e, i in idx.items()))
    for label, C0, n_iter in (("no seed, n_iter=3", None, 3), ("no seed, n_iter=6", None, 6),
                              ("seed=truth, n_iter=3", wmass, 3)):
        r_sa = saha_boltzmann_solve_np(tok_sa, fv, w, tab, C0=C0, T0=10000.0, log10_Ne0=17.0,
                                       log10_Nl0=np.log10(Nl), sa_correction=True, n_iter=n_iter)
        print(f"  self-absorbed, corrected ({label}): T={r_sa.T:.1f}, log10 Ne={r_sa.log10_Ne:.3f}; "
              + ", ".join(f"{e}={r_sa.mass_fractions[i]:.4f}" for e, i in idx.items()))
        if label.startswith("seed"):
            for e, i in idx.items():
                rel = abs(r_sa.mass_fractions[i] / comp[e] - 1)
                check(rel < 0.10, f"self-absorbed + correction ({label}): {e} within 10 % ({rel:.2%})")
            check(abs(r_sa.T / T_true - 1) < 0.02, f"self-absorbed + correction: T within 2 % ({abs(r_sa.T / T_true - 1):.2%})")
    # perturbed seed (what a seed head would give)
    rng = np.random.default_rng(0)
    C0p = wmass * np.exp(rng.normal(0.0, 0.3, wmass.size))
    C0p /= C0p.sum()
    r_p = saha_boltzmann_solve_np(tok_sa, fv, w, tab, C0=C0p, T0=10000.0, log10_Ne0=17.0,
                                  log10_Nl0=np.log10(Nl), sa_correction=True, n_iter=3)
    worst = max(abs(r_p.mass_fractions[i] / comp[e] - 1) for e, i in idx.items())
    check(worst < 0.10, f"self-absorbed + correction, seed perturbed ±30 %: majors within 10 % (worst {worst:.2%})")
    return tok, tok_sa, Nl


def check_torch_vs_numpy(names, tab, tokens_path: Path, tok_fallback, device):
    print("\n(iii) torch vs numpy")
    try:
        with h5py.File(tokens_path, "r") as f:
            tok = f["tokens"][:3].astype(np.float64)
            fv = f["fit_valid"][:3].astype(np.float64)
        print(f"  tokens from {tokens_path.name}: {tok.shape}, valid fits per spectrum {fv.sum(1).astype(int).tolist()}")
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN: cannot read {tokens_path} ({exc}); using synthetic tokens")
        tok = np.stack([tok_fallback] * 3)
        fv = np.ones(tok.shape[:2])
    B, L, _ = tok.shape
    rng = np.random.default_rng(1)
    w = rng.uniform(0.0, 1.0, (B, L))
    w[0, rng.uniform(size=L) < 0.3] = 0.0            # some exact zeros
    C0 = rng.uniform(0.0, 1.0, (B, len(names)))
    C0 /= C0.sum(1, keepdims=True)
    T0 = np.array([9000.0, 11000.0, 12500.0])
    lNe0 = np.array([16.8, 17.2, 17.5])
    lNl0 = np.array([15.5, 16.0, 16.5])
    tt = lambda a: torch.as_tensor(a, dtype=torch.float64, device=device)  # noqa: E731
    for sa in (False, True):
        layer = SahaBoltzmannLayer(tab, {"sa_correction": sa}).to(device)
        with torch.no_grad():
            out = layer(tt(tok), tt(fv), tt(w), tt(C0), tt(T0), tt(lNe0), tt(lNl0))
        worst = 0.0
        for b in range(B):
            r = saha_boltzmann_solve_np(tok[b], fv[b], w[b], tab, C0=C0[b], T0=T0[b], log10_Ne0=lNe0[b],
                                        log10_Nl0=lNl0[b], sa_correction=sa)
            pairs = {
                "T": (out["T"][b].item(), r.T),
                "log10_Ne": (out["log10_Ne"][b].item(), r.log10_Ne),
                "concentrations": (out["concentrations"][b].cpu().numpy(), r.mass_fractions),
                "number_fractions": (out["number_fractions"][b].cpu().numpy(), r.number_fractions),
                "tau0": (out["tau0"][b].cpu().numpy(), r.tau0),
                "intercepts": (np.nan_to_num(out["intercepts"][b].cpu().numpy()), np.nan_to_num(r.intercepts)),
            }
            for k, (a, ref) in pairs.items():
                a, ref = np.asarray(a, float), np.asarray(ref, float)
                rel = float(np.max(np.abs(a - ref) / np.maximum(np.abs(ref), 1e-300)))
                worst = max(worst, rel)
            rr = float(np.max(np.abs(out["resid"][b].cpu().numpy() - r.resid)))
            check(rr < 1e-8, f"sa={sa} b={b}: residuals equal (max abs {rr:.1e})")
            check(bool((out["n_lines_used"][b].cpu().numpy() == r.n_lines_used).all()), f"sa={sa} b={b}: n_lines_used equal")
            check(bool((out["censored"][b].cpu().numpy() == r.censored).all()), f"sa={sa} b={b}: censored equal")
            check(bool((out["used_mask"][b].cpu().numpy() == r.used_mask).all()), f"sa={sa} b={b}: used_mask equal")
            check(bool(np.isclose(out["concentrations"][b].sum().item(), 1.0)), f"sa={sa} b={b}: mass fractions sum to 1")
        check(worst < 1e-6, f"sa={sa}: torch == numpy on T, log10 Ne, concentrations, number fractions, tau0, intercepts (max rel {worst:.1e})")
    return tok, fv


def check_gradients(names, tab, tok_meas, fv_meas, tok_syn, device):
    print("\n(iv) gradients")
    tt = lambda a, g=False: torch.as_tensor(a, dtype=torch.float64, device=device).requires_grad_(g)  # noqa: E731
    B, L, _ = tok_meas.shape
    rng = np.random.default_rng(2)
    for sa in (False, True):
        layer = SahaBoltzmannLayer(tab, {"sa_correction": sa}).to(device)
        w = tt(rng.uniform(0.2, 1.0, (B, L)), True)
        T0 = tt(np.array([9000.0, 11000.0, 12500.0]), True)
        lNe0 = tt(np.array([16.8, 17.2, 17.5]), True)
        lNl0 = tt(np.array([15.5, 16.0, 16.5]), True)
        C0 = rng.uniform(0.0, 1.0, (B, len(names)))
        C0 /= C0.sum(1, keepdims=True)
        out = layer(tt(tok_meas), tt(fv_meas), w, tt(C0), T0, lNe0, lNl0)
        loss = (torch.log(out["concentrations"] + 1e-7).sum() + out["T"].sum() / 1e4
                + out["log10_Ne"].sum() + out["tau0"].sum() + out["resid"].pow(2).sum())
        grads = torch.autograd.grad(loss, [w, T0, lNe0, lNl0], allow_unused=True)
        for name, g in zip(["weights", "T0", "log10_Ne0", "log10_Nl0"], grads):
            ok = g is not None and bool(torch.isfinite(g).all())
            nz = float(g.abs().max()) if g is not None else 0.0
            check(ok, f"sa={sa}: finite gradient w.r.t. {name} (max |g| = {nz:.3g})")
        for k in ("concentrations", "T", "log10_Ne", "tau0", "resid", "number_fractions"):
            check(bool(torch.isfinite(out[k]).all()), f"sa={sa}: output {k} finite")

    # finite-difference Jacobian check on a tiny float64 problem: 2 elements, 14 synthetic lines.
    # torch.autograd.gradcheck's default eps=1e-6 is below the round-off noise of the
    # linear solve (~1e-13 relative on outputs of size 1e2), so central differences with
    # eps=1e-4 are compared against the analytic Jacobian on entries that are not tiny.
    # With sa_correction the curve-of-growth table is piecewise linear in ln(gamma/sigma),
    # so a slightly looser tolerance is used there.
    small_names = ["Fe", "Cr"]
    small_tab = build_cf_tables(small_names, tab.meta["db_path"])
    sel = np.concatenate([np.flatnonzero(tok_syn[:, 7] == 26)[:8], np.flatnonzero(tok_syn[:, 7] == 24)[:6]])
    tok_s = tt(tok_syn[sel][None])
    fv_s = tt(np.ones((1, sel.size)))
    C0_s = tt(np.array([[0.75, 0.25]]))
    # With the self-absorption correction the curve of growth comes from a
    # piecewise-linear table in ln(a) (T-dependent damping ratio); central
    # differences straddling a table knot disagree with the analytic slope by
    # a few percent on the T0 column, which is harmless for SGD. sa=False
    # exercises the exact path and keeps the strict tolerance.
    for sa, tol in ((False, 1e-3), (True, 5e-2)):
        layer = SahaBoltzmannLayer(small_tab, {"sa_correction": sa, "n_iter": 2}).to(device)

        def fn(w, T0, lNe0, lNl0):
            o = layer(tok_s, fv_s, w, C0_s, T0, lNe0, lNl0)
            return torch.stack([o["concentrations"][0, 0], o["T"][0] / 1e4, o["log10_Ne"][0],
                                o["tau0"][0].sum() / 1e2, o["resid"][0].pow(2).sum()])

        inputs = (tt(rng.uniform(0.3, 1.0, (1, sel.size)), True), tt([10500.0], True),
                  tt([17.1], True), tt([16.2], True))
        J = torch.autograd.functional.jacobian(fn, inputs)
        worst = 0.0
        for k, inp in enumerate(inputs):
            n_in = inp.numel()
            step = 1e-4 * (1.0 if k != 1 else 100.0)          # T0 in K
            num = np.zeros((5, n_in))
            for j in range(n_in):
                plus = [i.detach().clone() for i in inputs]
                minus = [i.detach().clone() for i in inputs]
                plus[k].reshape(-1)[j] += step
                minus[k].reshape(-1)[j] -= step
                with torch.no_grad():
                    num[:, j] = ((fn(*plus) - fn(*minus)) / (2 * step)).cpu().numpy()
            ana = J[k].reshape(5, -1).cpu().numpy()
            scale = np.maximum(np.abs(num).max(axis=1, keepdims=True), 1e-8)
            rel = np.abs(ana - num) / scale                    # relative to the largest entry per output
            worst = max(worst, float(rel.max()))
            check(np.isfinite(ana).all(), f"sa={sa}: analytic Jacobian w.r.t. input {k} finite")
        check(worst < tol, f"sa={sa}: analytic vs central-difference Jacobian (eps=1e-4) on (weights, T0, log10_Ne0, log10_Nl0): max rel {worst:.1e} < {tol:g}")


def check_degenerate(names, tab, tok_syn, device):
    print("\n(v) degenerate cases")
    iFe, iCr, iNi, iC = (names.index(e) for e in ("Fe", "Cr", "Ni", "C"))
    Z = tok_syn[:, 7]
    z = tok_syn[:, 8]
    L = tok_syn.shape[0]
    fv = np.ones(L)
    C0 = np.zeros(len(names))
    C0[[iFe, iCr, iNi, iC]] = [0.7, 0.18, 0.09, 0.03]

    # (a) Cr with a single line, Ni with only stage-II lines, C excluded
    w = np.ones(L)
    w[Z == 24] = 0.0
    w[np.flatnonzero(Z == 24)[0]] = 1.0
    w[(Z == 28) & (z == 0)] = 0.0
    w[Z == 6] = 0.0
    # One identified line is enough (production default): the element takes the
    # common Fe-dominated slope and joins the closure sum.
    r = saha_boltzmann_solve_np(tok_syn, fv, w, tab, C0=C0, T0=10000.0, log10_Ne0=17.0, log10_Nl0=16.4)
    finite = np.isfinite(r.mass_fractions).all() and np.isfinite(r.T) and np.isfinite(r.log10_Ne) and np.isfinite(r.tau0).all()
    check(finite, "single-line Cr / ion-only Ni / seeded C: all outputs finite")
    check(r.n_lines_used[iCr] == 1 and r.source[iCr] == "cf", f"single-line element solved (Cr = {r.mass_fractions[iCr]:.4f}, true 0.18)")
    check(r.n_lines_used[iNi] > 0 and np.all(z[(Z == 28) & r.used_mask] == 1), f"stage-II-only element solved (Ni = {r.mass_fractions[iNi]:.4f}, true 0.09)")
    check(r.n_lines_used[iC] == 0 and r.source[iC] == "none" and np.isnan(r.intercepts[iC])
          and r.mass_fractions[iC] == 0.0,
          f"element without any line: not summed, source 'none' (C = {r.mass_fractions[iC]:.4f})")
    # legacy rule, still reachable: the seed keeps its mass in the closure
    r_seed = saha_boltzmann_solve_np(tok_syn, fv, w, tab, C0=C0, T0=10000.0, log10_Ne0=17.0,
                                     log10_Nl0=16.4, seed_in_closure=True)
    check(r_seed.source[iC] == "seed" and r_seed.mass_fractions[iC] > 0.0,
          f"seed_in_closure=True: element without lines keeps C0 (C = {r_seed.mass_fractions[iC]:.4f} from C0 0.03)")
    check(abs(r.mass_fractions[iFe] / 0.7 - 1) < 0.1, f"Fe still within 10 % ({r.mass_fractions[iFe]:.4f})")
    check(abs(r.mass_fractions.sum() - 1) < 1e-9, "mass fractions sum to 1")

    # (b) all weights zero
    r0 = saha_boltzmann_solve_np(tok_syn, fv, np.zeros(L), tab, C0=C0, T0=10000.0, log10_Ne0=17.0, log10_Nl0=16.4)
    check(np.isfinite(r0.mass_fractions).all() and np.isfinite(r0.T) and np.isfinite(r0.tau0).all(), "all weights zero: finite outputs")
    check(np.all(r0.source == "seed") and np.allclose(r0.mass_fractions, C0), "all weights zero: every element from the seed, concentrations == C0")
    check(abs(r0.T - 10000.0) < 1.0 and abs(r0.log10_Ne - 17.0) < 1e-3, f"all weights zero: T, Ne fall back to the priors ({r0.T:.1f} K, {r0.log10_Ne:.3f})")
    check(bool(np.all(r0.censored == (C0 < tab.lod))), "all weights zero: censored == (C0 < LOD)")
    r00 = saha_boltzmann_solve_np(tok_syn, fv, np.zeros(L), tab, C0=None)
    check(np.all(r00.source == "none") and np.all(r00.mass_fractions == 0) and np.all(r00.censored), "all weights zero, no C0: source 'none', zeros, censored")

    # (c) torch on the same cases (batch of 3), no NaNs, differentiable
    tt = lambda a: torch.as_tensor(a, dtype=torch.float64, device=device)  # noqa: E731
    layer = SahaBoltzmannLayer(tab).to(device)
    W = torch.stack([tt(w), tt(np.zeros(L)), tt(np.ones(L))]).requires_grad_(True)
    out = layer(tt(np.stack([tok_syn] * 3)), tt(np.ones((3, L))), W, tt(np.stack([C0] * 3)))
    allfinite = all(bool(torch.isfinite(out[k]).all()) for k in ("concentrations", "T", "log10_Ne", "tau0", "resid"))
    check(allfinite, "torch degenerate batch: finite outputs")
    check(bool(torch.allclose(out["concentrations"][1], tt(C0))) and bool((~out["has_lines"][1]).all()), "torch all-zero-weight row == C0")
    g = torch.autograd.grad(out["concentrations"].sum() + out["T"].sum(), W)[0]
    check(bool(torch.isfinite(g).all()), "torch degenerate batch: finite gradient")
    check(bool((out["intercepts"][1].isnan()).all()) and bool((~out["intercepts"][2].isnan())[[iFe, iCr, iNi, iC]].all()),
          "torch intercepts nan exactly where no line is used")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--measured_cache", default=str(DEFAULT_MEASURED))
    ap.add_argument("--tokens", default=str(DEFAULT_TOKENS))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.set_default_dtype(torch.float64)

    names = element_names_from_cache(Path(args.measured_cache))
    tab = check_tables(names, args.db)
    check_cog(tab)
    tok_syn, tok_sa, _ = check_round_trip(names, tab, args.db)
    tok_meas, fv_meas = check_torch_vs_numpy(names, tab, Path(args.tokens), tok_syn, device)
    check_gradients(names, tab, tok_meas, fv_meas, tok_syn, device)
    check_degenerate(names, tab, tok_syn, device)

    print()
    if _failures:
        print(f"FAILED: {len(_failures)} check(s)")
        for m in _failures:
            print(f"  - {m}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
