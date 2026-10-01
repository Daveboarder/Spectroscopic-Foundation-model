"""
Physics and plumbing checks for the physics_version-2 generator
(`data/two_zone_pipeline.py` on top of `data/plasma_physics.py`).

Checks (each prints PASS / FAIL / SKIP and a number):
  (a) Kirchhoff: emissivity / absorption == line source function (1e-10);
      optically thin limit: integral of the emergent line == eps_int * l (1e-4)
  (b) optically thick limit: line-centre radiance -> source function (1e-4)
      and -> Planck B_lambda(T) (1e-2, level-energy rounding)
  (c) identical zones with l_inner + 2 l_outer == l reproduce one zone (1e-6)
  (d) self-reversal of strong Fe I resonance lines for a hot core / cold shell
      (centre / horn < 0.8)
  (e) round trip through cf.solver_np.saha_boltzmann_solve_np on a thin
      one-zone shot with exact areas (T +-1 %, log10 Ne +-0.04, majors +-5 %);
      skipped when the cf package is not importable yet
  (f) fine-grid convergence: halving fine_step_nm changes the unit-normalised
      spectrum by < 1e-3
  (g) end-to-end smoke dataset from --libs_data_config: shapes, contract-C1
      columns, extract_plasma_targets; measured cache table -> has_plasma_labels 0,
      measured_groups / grouped splits.

Outputs land in `sanity_checks/two_zone_<timestamp>/` (report.txt + figures).
Exit code 0 when every non-skipped check passes, 2 otherwise.

Usage:
    uv run python scripts/check_two_zone_physics.py
    uv run python scripts/check_two_zone_physics.py --libs_data_config config/libs_data_cf_smoke.yaml
    uv run python scripts/check_two_zone_physics.py --skip_dataset
"""

from __future__ import annotations

import argparse
import glob
import json
import os
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import plasma_physics as pp
from data.atomic_data import mass_to_number_fractions, number_to_mass_fractions
from data.libs_pipeline import (
    ZONE_COLUMNS,
    build_dataset_from_config,
    extract_finetune_labels,
    extract_plasma_targets,
    get_or_make_splits,
    load_wavelength,
    make_group_splits,
    measured_groups,
    unit_norm,
)
from data.line_tokenization import N_FEATURES, atomic_number, ion_binary
from data.two_zone_pipeline import (
    ZoneState,
    generate_zone_sample_table,
    instrument_kernel,
    make_fine_grid,
    synthesise_fine_grid,
    synthesise_spectrum,
    zone_states_from_row,
)
from data.two_zone_pipeline import _number_density_rows

DB_DEFAULT = "external_data/Source/LIBS_data.db"


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []   # (check, status, detail)

    def add(self, name: str, ok: bool | None, detail: str) -> None:
        status = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
        self.rows.append((name, status, detail))
        print(f"  [{status}] {name}: {detail}")

    @property
    def failed(self) -> bool:
        return any(s == "FAIL" for _, s, _ in self.rows)

    def table(self) -> str:
        w = max(len(r[0]) for r in self.rows) if self.rows else 10
        lines = [f"{'check':<{w}}  status  detail", "-" * (w + 40)]
        for name, status, detail in self.rows:
            lines.append(f"{name:<{w}}  {status:<6}  {detail}")
        lines.append("")
        lines.append("OVERALL: " + ("FAIL" if self.failed else "PASS"))
        return "\n".join(lines)


def _quasi_neutral_N(elements, x, T, Ne, db_path) -> float:
    return float(_number_density_rows(elements, np.asarray([x]), np.asarray([T]), np.asarray([Ne]), db_path, None)[0])


def _fe_resonance_lines(db_path: str, T: float, lo: float, hi: float, n: int = 5) -> np.ndarray:
    """Strongest Fe I lines with E_i < 0.2 eV inside [lo, hi] nm (by thin emissivity)."""
    ls = pp.line_set_for_element("Fe", T, 1e17, db_path)
    sel = ls.is_I & (ls.Ei < 0.2) & (ls.wl_nm > lo) & (ls.wl_nm < hi)
    order = np.argsort(-ls.eps_per_n * sel)
    return order[:n]


# ─────────────────────────────────────────────────────────────────────────────
# (a) Kirchhoff + thin limit
# ─────────────────────────────────────────────────────────────────────────────
def check_kirchhoff_and_thin(rep: Report, db_path: str) -> None:
    T, Ne = 10000.0, 1e17
    ls = pp.line_set_for_element("Fe", T, Ne, db_path)
    frac = ls.stage_fraction()
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = ls.eps_per_n / (ls.kt * frac)
    rel = np.abs(ratio / ls.S - 1.0)
    rel = rel[np.isfinite(rel)]
    rep.add("(a) Kirchhoff eps/kappa == S", float(rel.max()) < 1e-10,
            f"max |eps/(kappa) / S - 1| = {rel.max():.2e} over {rel.size} Fe lines")

    # thin limit on a small axis, Gaussian-only profile, fine grid 0.0005 nm
    axis = np.linspace(245.0, 255.0, 400)
    grid = make_fine_grid(axis, 0.0005)
    N, l = 1e8, 0.1                      # tau0 ~ 1e-9 for the strongest line
    inner = ZoneState(T=T, Ne=Ne, N=N, l=l, gamma_nm=0.0)
    rad, lines = synthesise_fine_grid(["Fe"], np.array([1.0]), grid, inner, None, db_path,
                                      line_window_nm=0.4, min_relative_intensity=0.0,
                                      adaptive_window=False, return_lines=True)
    integral = np.trapezoid(rad, grid) * pp.NM_TO_CM
    expected = float(np.sum(lines[0]["I_thin"]))
    tau_max = float(lines[0]["tau0"].max())
    err = abs(integral / expected - 1.0)
    rep.add("(a) thin limit int I dl == eps*l", err < 1e-4,
            f"rel err {err:.2e} (max tau0 {tau_max:.1e}, {lines[0]['wl'].size} lines)")


# ─────────────────────────────────────────────────────────────────────────────
# (b) thick limit
# ─────────────────────────────────────────────────────────────────────────────
def check_thick_limit(rep: Report, db_path: str) -> None:
    T, Ne = 10000.0, 1e17
    axis = np.linspace(247.0, 250.0, 300)
    grid = make_fine_grid(axis, 0.001)
    inner = ZoneState(T=T, Ne=Ne, N=1e20, l=10.0, gamma_nm=0.01)
    rad, lines = synthesise_fine_grid(["Fe"], np.array([1.0]), grid, inner, None, db_path,
                                      line_window_nm=0.4, min_relative_intensity=0.0,
                                      adaptive_window=True, return_lines=True)
    ln = lines[0]
    k = int(np.argmax(ln["tau0"]))
    wl0 = float(ln["wl"][k])
    S_line = float(pp.line_source_function(wl0, ln["Ei"][k], ln["Ek"][k], T))
    B = float(pp.planck_lambda(wl0, T))
    i0 = float(np.interp(wl0, grid, rad))
    err_S = abs(i0 / S_line - 1.0)
    err_B = abs(i0 / B - 1.0)
    rep.add("(b) thick limit I(centre) -> S_line", err_S < 1e-4,
            f"Fe I {wl0:.3f} nm, tau0 {ln['tau0'][k]:.1e}: I/S - 1 = {err_S:.2e}")
    rep.add("(b) thick limit I(centre) -> B_lambda(T)", err_B < 1e-2,
            f"I/B_lambda - 1 = {err_B:.2e} (level-energy vs h c/lambda rounding)")


# ─────────────────────────────────────────────────────────────────────────────
# (c) identical zones == one zone
# ─────────────────────────────────────────────────────────────────────────────
def check_identical_zones(rep: Report, db_path: str) -> None:
    T, Ne = 12000.0, 5e17
    elements = ["Fe", "Mn", "Cr"]
    x = np.array([0.9, 0.06, 0.04])
    N = _quasi_neutral_N(elements, x, T, Ne, db_path)
    axis = np.linspace(255.0, 265.0, 400)
    grid = make_fine_grid(axis, 0.002)
    l_in, l_out = 0.1, 0.05
    one = ZoneState(T, Ne, N, l_in + 2 * l_out, 0.01)
    inner = ZoneState(T, Ne, N, l_in, 0.01)
    outer = ZoneState(T, Ne, N, l_out, 0.01)
    kw = dict(line_window_nm=0.4, min_relative_intensity=1e-7, adaptive_window=False)
    r1 = synthesise_fine_grid(elements, x, grid, one, None, db_path, **kw)
    r2 = synthesise_fine_grid(elements, x, grid, inner, outer, db_path, **kw)
    err = float(np.max(np.abs(r1 - r2)) / np.max(r1))
    rep.add("(c) identical zones == one zone", err < 1e-6,
            f"max |two - one| / max = {err:.2e} (N = {N:.2e} cm^-3)")


# ─────────────────────────────────────────────────────────────────────────────
# (d) self-reversal
# ─────────────────────────────────────────────────────────────────────────────
def check_self_reversal(rep: Report, db_path: str, out_dir: Path) -> None:
    """Hot core / cold shell on pure Fe. Two statements are tested:
    1. at the prescribed state (quasi-neutral N2, l_outer 0.05 cm) the shell
       absorbs the core of every strong Fe I resonance line:
       I_two(centre) / I_one(centre) < 0.8;
    2. with the shell thinned to tau_outer(centre) = 10 for the strongest line
       the classic self-reversed shape appears: centre / horn < 0.8 with the
       horns inside +-0.3 nm.
    At the prescribed state the quasi-neutral shell (N2 ~ 4e19 cm^-3) is so
    thick (tau ~ 1e5) that the whole window is black, so statement 2 needs
    the thinner shell."""
    Te1, Ne1, Te2, Ne2 = 15000.0, 3e17, 4000.0, 3e15
    l_inner, l_outer = 0.2, 0.05
    gamma = 0.01
    elements, x = ["Fe"], np.array([1.0])
    N1 = _quasi_neutral_N(elements, x, Te1, Ne1, db_path)
    N2 = _quasi_neutral_N(elements, x, Te2, Ne2, db_path)
    axis = load_wavelength("external_data/Data/VASKUT K8.json")
    grid = make_fine_grid(axis, 0.002)
    inner = ZoneState(Te1, Ne1, N1, l_inner, gamma)
    outer = ZoneState(Te2, Ne2, N2, l_outer, gamma)

    ls1 = pp.line_set_for_element("Fe", Te1, Ne1, db_path)
    ls2 = pp.line_set_for_element("Fe", Te2, Ne2, db_path)
    idx = _fe_resonance_lines(db_path, Te1, float(axis.min()) + 1, float(axis.max()) - 1, n=5)
    # shell optical depth at line centre for the strongest resonance line
    k0 = idx[0]
    sig2 = float(pp.doppler_sigma_nm(ls2.wl_nm[k0], Te2, 55.845))
    tau2_per_cm = float(ls2.kappa_int(N2)[k0] * pp.voigt_peak(sig2, gamma) / pp.NM_TO_CM)
    tau2_0 = tau2_per_cm * l_outer
    l_thin = 10.0 / tau2_per_cm
    outer_thin = ZoneState(Te2, Ne2, N2, l_thin, gamma)

    rad1 = synthesise_fine_grid(elements, x, grid, inner, None, db_path)
    rad2 = synthesise_fine_grid(elements, x, grid, inner, outer, db_path)
    rad3 = synthesise_fine_grid(elements, x, grid, inner, outer_thin, db_path)

    core, horn, msgs_core, msgs_horn = [], [], [], []
    fig, axes = plt.subplots(2, len(idx), figsize=(3.2 * len(idx), 5.4))
    for col, k in enumerate(idx):
        wl0 = float(ls1.wl_nm[k])
        win = (grid > wl0 - 0.3) & (grid < wl0 + 0.3)
        c1, c2, c3 = (float(np.interp(wl0, grid, r)) for r in (rad1, rad2, rad3))
        core.append(c2 / c1)
        h3 = float(rad3[win].max())
        horn.append(c3 / h3 if h3 > 0 else np.nan)
        msgs_core.append(f"{wl0:.3f}:{c2 / c1:.2f}")
        msgs_horn.append(f"{wl0:.3f}:{c3 / h3:.2f}")
        ax = axes[0, col]
        ax.plot(grid[win], rad1[win], lw=0.8, color="grey", label="one zone")
        ax.plot(grid[win], rad2[win], lw=0.8, color="crimson", label=f"two zone, l_out {l_outer} cm")
        ax.set_title(f"Fe I {wl0:.3f}  core two/one = {c2 / c1:.2f}", fontsize=8)
        ax = axes[1, col]
        ax.plot(grid[win], rad1[win], lw=0.8, color="grey")
        ax.plot(grid[win], rad3[win], lw=0.8, color="navy", label=f"two zone, l_out {l_thin:.1e} cm")
        ax.set_title(f"tau_out(0)=10: centre/horn = {c3 / h3:.2f}", fontsize=8)
        for a in axes[:, col]:
            a.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=7)
    axes[1, 0].legend(fontsize=7)
    fig.suptitle(f"Self-reversal, pure Fe: Te1 {Te1:.0f} K / Ne1 {Ne1:.0e} (N1 {N1:.1e})  |  "
                 f"Te2 {Te2:.0f} K / Ne2 {Ne2:.0e} (N2 {N2:.1e} cm^-3, tau_out(0) {tau2_0:.1e} at l_out {l_outer})",
                 fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "self_reversal.png", dpi=130)
    plt.close(fig)
    core = np.asarray(core)
    horn = np.asarray(horn)
    rep.add("(d) self-reversal: shell absorbs core", bool(np.all(core < 0.8)),
            f"I_two/I_one at centre = [{', '.join(msgs_core)}]  (N1 {N1:.2e}, N2 {N2:.2e} cm^-3, "
            f"shell tau0 {tau2_0:.1e} for Fe I {ls1.wl_nm[k0]:.3f})")
    rep.add("(d) self-reversal: horns for tau_out(0) = 10", bool(np.all(horn < 0.8)),
            f"centre/horn(+-0.3 nm) = [{', '.join(msgs_horn)}]  (l_outer {l_thin:.2e} cm)")


# ─────────────────────────────────────────────────────────────────────────────
# (e) CF round trip (optional)
# ─────────────────────────────────────────────────────────────────────────────
def _thin_tokens(elements: list[str], x: np.ndarray, T: float, Ne: float, N: float, l: float,
                 db_path: str, lo: float, hi: float) -> np.ndarray:
    rows = []
    for j, elem in enumerate(elements):
        if x[j] <= 0:
            continue
        d = pp.thin_line_intensities(elem, T, Ne, float(x[j]), N, l, db_path)
        sel = (d["wl"] > lo) & (d["wl"] < hi) & (d["I_int"] > 0)
        Z = atomic_number(elem)
        for i in np.nonzero(sel)[0]:
            tok = np.zeros(N_FEATURES, dtype=np.float64)
            tok[0] = d["wl"][i]; tok[1] = d["Ei"][i]; tok[2] = d["Ek"][i]
            tok[3] = np.log10(d["gi"][i]); tok[4] = np.log10(d["gk"][i]); tok[5] = np.log10(d["Ak"][i])
            tok[6] = np.log10(d["I_int"][i]); tok[7] = Z; tok[8] = ion_binary(d["ion_state"][i])
            tok[9] = d["I_int"][i]; tok[10] = 0.05; tok[11] = 1.0; tok[12] = 0.0; tok[13] = 0.0
            rows.append(tok)
    return np.asarray(rows, dtype=np.float64)


def check_cf_round_trip(rep: Report, db_path: str) -> None:
    try:
        from cf.solver_np import saha_boltzmann_solve_np
        from cf.tables import build_cf_tables
    except Exception as exc:  # package not there yet
        rep.add("(e) CF round trip (cf.solver_np)", None, f"cf package not importable ({exc.__class__.__name__}: {exc})")
        return
    elements = ["Fe", "Mn", "Cr", "Ni", "Cu", "Si", "C", "Al"]
    w_true = np.array([0.90, 0.03, 0.02, 0.02, 0.01, 0.01, 0.005, 0.005])
    w_true /= w_true.sum()
    x_true = mass_to_number_fractions(w_true, elements)
    T, Ne = 11000.0, 2e17
    N = 1e8                               # optically thin by construction
    l = 0.1
    tokens = _thin_tokens(elements, x_true, T, Ne, N, l, db_path, 150.0, 415.0)
    # keep the 400 strongest lines so the solve stays quick and well conditioned
    keep = np.argsort(-tokens[:, 9])[:400]
    tokens = tokens[keep]
    L = tokens.shape[0]
    fit_valid = np.ones(L, dtype=bool)
    weights = np.ones(L, dtype=np.float64)
    try:
        tables = build_cf_tables(elements, db_path)
        res = saha_boltzmann_solve_np(tokens, fit_valid, weights, tables, C0=None,
                                      T0=10000.0, log10_Ne0=17.0, log10_Nl0=np.log10(N * l),
                                      n_iter=3, sa_correction=True)
    except Exception as exc:
        rep.add("(e) CF round trip (cf.solver_np)", None, f"solver raised {exc.__class__.__name__}: {exc}")
        return
    t_err = abs(float(res.T) / T - 1.0)
    ne_err = abs(float(res.log10_Ne) - np.log10(Ne))
    w_pred = np.asarray(res.mass_fractions, dtype=np.float64)
    major = w_true >= 0.01
    c_err = np.abs(w_pred[major] / w_true[major] - 1.0)
    ok = (t_err < 0.01) and (ne_err < np.log10(1.1)) and bool(np.all(c_err < 0.05))
    rep.add("(e) CF round trip (cf.solver_np)", ok,
            f"T {res.T:.0f} K (true {T:.0f}, {100 * t_err:.2f} %), log10 Ne {res.log10_Ne:.3f} "
            f"(true {np.log10(Ne):.3f}), majors max rel err {100 * c_err.max():.2f} % "
            f"[{', '.join(f'{e}:{p:.4f}/{t:.4f}' for e, p, t in zip(elements, w_pred, w_true) if p > 0 or t > 0)}]")


# ─────────────────────────────────────────────────────────────────────────────
# (f) fine-grid convergence
# ─────────────────────────────────────────────────────────────────────────────
def check_grid_convergence(rep: Report, db_path: str, gen_cfg: dict) -> None:
    """Halving fine_step_nm must change the unit-normalised spectrum by < 1e-3
    at the production step (0.002 nm). The config step is reported as well
    when it differs (smoke configs use a coarser grid on purpose)."""
    axis = load_wavelength("external_data/Data/VASKUT K8.json")
    elements = ["Fe", "Mn", "Cr", "Ni", "Si", "C"]
    w = np.array([0.93, 0.02, 0.02, 0.015, 0.01, 0.005])
    zone_cfg = gen_cfg.get("zones", {})
    tab = generate_zone_sample_table(
        {e: (float(v), float(v)) for e, v in zip(elements, w)}, 2, "CONV", "conv",
        zone_cfg, np.random.default_rng(7), two_zone_fraction=1.0, number_density="auto", db_path=db_path,
    )
    cfg_step = float(gen_cfg.get("fine_step_nm", 0.002))
    steps = [0.002] if abs(cfg_step - 0.002) < 1e-12 else [0.002, cfg_step]
    results = {}
    for step in steps:
        worst = 0.0
        for i in range(len(tab)):
            row = tab.iloc[i].to_dict()
            s_coarse = synthesise_spectrum(elements, w, axis, row, db_path, {**gen_cfg, "fine_step_nm": step})
            s_fine = synthesise_spectrum(elements, w, axis, row, db_path, {**gen_cfg, "fine_step_nm": step / 2})
            worst = max(worst, float(np.max(np.abs(s_coarse - s_fine))))
        results[step] = worst
    detail = ", ".join(f"max |spec({st}) - spec({st / 2})| = {v:.2e}" for st, v in results.items())
    rep.add("(f) fine-grid convergence (step 0.002)", results[0.002] < 1e-3,
            detail + " after unit_norm (2 two-zone shots)")


# ─────────────────────────────────────────────────────────────────────────────
# (g) smoke dataset + plumbing
# ─────────────────────────────────────────────────────────────────────────────
def _read_cache_table(path: str) -> pd.DataFrame:
    with h5py.File(path, "r") as f:
        grp = f["sample_table"]
        cols = json.loads(grp.attrs["columns"])
        data = {}
        for c in cols:
            v = grp[c][:]
            if v.dtype.kind in ("S", "O"):
                v = [s.decode("utf-8") if isinstance(s, bytes) else s for s in v]
            data[c] = v
    return pd.DataFrame(data)


def check_dataset(rep: Report, cfg: dict, cfg_path: str, out_dir: Path) -> None:
    t0 = time.time()
    ds = build_dataset_from_config(cfg)
    dt = time.time() - t0
    table, spectra = ds.sample_table, ds.spectra
    n_expected = cfg["generation"]["max_sample_types"] * cfg["generation"]["n_samples_per_type"]
    ok_shape = spectra.shape == (n_expected, ds.wavelength.size) and len(table) == n_expected
    rep.add("(g) smoke dataset shape", ok_shape,
            f"spectra {spectra.shape}, table {len(table)} rows, built/loaded in {dt:.1f} s "
            f"(cache key {ds.cache_key})")

    missing = [c for c in ZONE_COLUMNS + ("Te", "Ne") if c not in table.columns]
    rep.add("(g) contract C1 columns", not missing, "all present" if not missing else f"missing {missing}")
    if missing:
        return
    conc, elem_names, _ = extract_finetune_labels(table)
    sums = conc.sum(axis=1)
    ok_conc = np.allclose(sums, 1.0, atol=1e-5) and not any(c in elem_names for c in ZONE_COLUMNS)
    rep.add("(g) element columns are mass fractions summing to 1", ok_conc,
            f"{len(elem_names)} elements, sums in [{sums.min():.6f}, {sums.max():.6f}]")

    pm = table["plasma_model"].astype(str).to_numpy()
    one = pm == "one_zone"
    ok_pm = set(np.unique(pm)) <= {"one_zone", "two_zone"}
    ok_alias = np.allclose(table["Te"], table["Te1"]) and np.allclose(table["Ne"], table["Ne1"])
    ok_one = (np.allclose(table.loc[one, "Te2"], table.loc[one, "Te1"])
              and np.allclose(table.loc[one, "Ne2"], table.loc[one, "Ne1"])
              and np.all(table.loc[one, "l_outer"] == 0.0)
              and np.allclose(table.loc[one, "N2"], table.loc[one, "N1"]))
    zc = cfg["generation"]["zones"]
    in_range = (table["Te1"].between(*zc["te1"]).all() and table["Ne1"].between(*zc["ne1"]).all()
                and table["l_inner"].between(*zc["l_inner_cm"]).all()
                and table["gamma_stark1"].between(*zc["gamma_stark_nm"]).all()
                and (table["N1"] > 0).all() and (table["N2"] > 0).all())
    rep.add("(g) plasma columns consistent", bool(ok_pm and ok_alias and ok_one and in_range),
            f"two_zone {int((~one).sum())}/{len(pm)}, aliases {ok_alias}, one_zone collapse {ok_one}, "
            f"ranges {in_range}; N1 [{table['N1'].min():.2e}, {table['N1'].max():.2e}], "
            f"N2 [{table['N2'].min():.2e}, {table['N2'].max():.2e}] cm^-3")

    ok_spec = (spectra.min() >= 0.0) and np.allclose(spectra.max(axis=1), 1.0) and np.all(np.isfinite(spectra))
    rep.add("(g) spectra unit-normalised and finite", bool(ok_spec),
            f"min {spectra.min():.3f}, per-shot max in [{spectra.max(axis=1).min():.3f}, {spectra.max(axis=1).max():.3f}]")

    tg = extract_plasma_targets(table)
    ok_tg = (all(v.dtype == np.float32 and v.shape == (len(table),) for v in tg.values())
             and np.all(tg["has_plasma_labels"] == 1)
             and np.array_equal(tg["is_two_zone"], (~one).astype(np.float32))
             and np.allclose(tg["Te"], table["Te1"].to_numpy(dtype=np.float32))
             and np.allclose(tg["log10_Nl"], np.log10(table["N1"] * table["l_inner"]).to_numpy(dtype=np.float32), atol=1e-5))
    rep.add("(g) extract_plasma_targets (synthetic)", bool(ok_tg),
            f"keys {sorted(tg)}; log10_Nl in [{tg['log10_Nl'].min():.2f}, {tg['log10_Nl'].max():.2f}]")

    # grouped split on the smoke table (writes splits_<key>_group_sample.json next to the cache)
    try:
        groups = measured_groups(table, by="sample")
        splits, path = get_or_make_splits(len(table), ds.cache_dir, ds.cache_key, 0.2, 0.2, 42,
                                          groups=groups, strategy="group_sample")
        leak = [k for k in ("val", "test") if set(groups[splits[k]]) & set(groups[splits["train"]])]
        rep.add("(g) get_or_make_splits(group_sample)", not leak and os.path.basename(path).endswith("_group_sample.json"),
                f"train/val/test = {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])} -> {os.path.basename(path)}")
    except Exception as exc:
        rep.add("(g) get_or_make_splits(group_sample)", False, f"{exc.__class__.__name__}: {exc}")

    # figure: first spectrum of each type
    fig, axes = plt.subplots(3, 1, figsize=(13, 7), sharex=True)
    for ax, i in zip(axes, range(0, len(table), cfg["generation"]["n_samples_per_type"])):
        r = table.iloc[i]
        ax.plot(ds.wavelength, spectra[i], lw=0.5)
        ax.set_title(f"{r['sample_type_name']}  {r['plasma_model']}  Te1 {r['Te1']:.0f} K  Ne1 {r['Ne1']:.1e}  "
                     f"l_in {r['l_inner']:.2f} cm  l_out {r['l_outer']:.2f} cm", fontsize=8, loc="left")
        ax.set_ylim(-0.02, 1.05)
    axes[-1].set_xlabel("wavelength (nm)")
    fig.tight_layout()
    fig.savefig(out_dir / "smoke_spectra.png", dpi=130)
    plt.close(fig)

    # measured cache
    cache_dir = ds.cache_dir
    files = sorted(glob.glob(os.path.join(cache_dir, "measured_cache_*.h5")), key=os.path.getmtime)
    if not files:
        rep.add("(g) extract_plasma_targets (measured)", None, "no measured_cache_*.h5 in cache dir")
        return
    mt = _read_cache_table(files[-1])
    tg = extract_plasma_targets(mt)
    ok_m = np.all(tg["has_plasma_labels"] == 0) and all(np.all(v == 0) for v in tg.values())
    rep.add("(g) extract_plasma_targets (measured)", bool(ok_m),
            f"{os.path.basename(files[-1])}: {len(mt)} rows, has_plasma_labels all 0 = {bool(np.all(tg['has_plasma_labels'] == 0))}")
    gs = measured_groups(mt, by="sample")
    gi = measured_groups(mt, by="instrument")
    ok_g = ("unknown" not in set(gi)) and len(np.unique(gi)) > 1 and len(np.unique(gs)) > 1
    try:
        sp_s = make_group_splits(gs, 0.15, 0.15, 42)
        sp_i = make_group_splits(gi, 0.15, 0.15, 42)
        leak_s = set(gs[sp_s["test"]]) & set(gs[sp_s["train"]])
        leak_i = set(gi[sp_i["test"]]) & set(gi[sp_i["train"]])
        ok_split = not leak_s and not leak_i
        detail = (f"{len(np.unique(gs))} samples, {len(np.unique(gi))} instruments {sorted(set(gi))[:4]}...; "
                  f"group_sample test {len(sp_s['test'])} rows / group_instrument test {len(sp_i['test'])} rows "
                  f"({sorted(set(gi[sp_i['test']]))}), no leakage {ok_split}")
    except Exception as exc:
        ok_split = False
        detail = f"{exc.__class__.__name__}: {exc}"
    rep.add("(g) measured_groups + grouped splits", bool(ok_g and ok_split), detail)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--libs_data_config", default="config/libs_data_cf_smoke.yaml")
    p.add_argument("--skip_dataset", action="store_true", help="skip the end-to-end smoke dataset build")
    p.add_argument("--out_dir", default="sanity_checks")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.libs_data_config))
    db_path = str(Path(cfg["paths"]["db"]).resolve())
    gen_cfg = cfg.get("generation", {})
    out_dir = Path(args.out_dir) / datetime.now().strftime("two_zone_%Y-%m-%d_%H-%M-%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Two-zone generator physics checks")
    print("=" * 70)
    print(f"Config: {args.libs_data_config}   DB: {db_path}")
    rep = Report()
    timings: dict[str, float] = {}

    for name, fn in [
        ("a", lambda: check_kirchhoff_and_thin(rep, db_path)),
        ("b", lambda: check_thick_limit(rep, db_path)),
        ("c", lambda: check_identical_zones(rep, db_path)),
        ("d", lambda: check_self_reversal(rep, db_path, out_dir)),
        ("e", lambda: check_cf_round_trip(rep, db_path)),
        ("f", lambda: check_grid_convergence(rep, db_path, gen_cfg)),
    ]:
        t0 = time.time()
        try:
            fn()
        except Exception as exc:
            rep.add(f"({name})", False, f"raised {exc.__class__.__name__}: {exc}")
        timings[name] = time.time() - t0

    if not args.skip_dataset:
        t0 = time.time()
        try:
            check_dataset(rep, cfg, args.libs_data_config, out_dir)
        except Exception as exc:
            rep.add("(g) dataset", False, f"raised {exc.__class__.__name__}: {exc}")
        timings["g"] = time.time() - t0

    print()
    print(rep.table())
    print("timings (s): " + ", ".join(f"{k} {v:.1f}" for k, v in timings.items()))
    (out_dir / "report.txt").write_text(rep.table() + "\n\ntimings: " + json.dumps(timings) + "\n", encoding="utf-8")
    print(f"[report] {out_dir / 'report.txt'}")
    sys.exit(2 if rep.failed else 0)


if __name__ == "__main__":
    main()
