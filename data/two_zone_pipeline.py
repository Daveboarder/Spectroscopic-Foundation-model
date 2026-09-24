"""
Physics-version-2 synthetic LIBS generator: one- and two-zone LTE radiative
transfer with physical optical depths.

Replaces the legacy ``data/libs_pipeline.create_spectra`` path (kept intact
for the old caches) when ``generation.plasma_model`` in the data YAML is one
of ``one_zone | two_zone | mixed``.  All line physics comes from
:mod:`data.plasma_physics` (Kirchhoff-consistent, CGS); this module only adds

* the per-shot plasma-state draw (:func:`generate_zone_sample_table`),
* the wavelength-resolved forward model (:func:`synthesise_spectrum`),
* the cache-aware dataset wrapper (:class:`TwoZoneSyntheticDataset`).

Forward model
-------------
Composition (mass fractions from the sample matrix) -> number fractions ``x``.
Zone 1 (inner, hot: ``Te1, Ne1, N1, l_inner``) and, for two-zone shots, zone 2
(outer, cooler: ``Te2, Ne2, N2, l_outer``) with the heavy-particle density
``N`` per zone from quasi-neutrality (``plasma_physics.number_density_from_ne``)
or a fixed value.  For every element and zone the DB lines give

    kappa_z(lambda) = sum_i kt_i(T_z) * x_e * N_z * r_stage,i * phi_i,z(lambda)   [cm^-1]
    eps_z(lambda)   = sum_i eps_i(T_z)  * x_e * N_z * phi_i,z(lambda)             [erg s^-1 cm^-3 sr^-1 cm^-1]
    S_z(lambda)     = eps_z / kappa_z  (== B_lambda(T_z) line by line, Kirchhoff)
    tau_z(lambda)   = kappa_z(lambda) * l_z

with ``phi_i,z`` an area-normalised Voigt profile (thermal Doppler sigma from
the atomic mass + a per-zone Lorentzian HWHM ``gamma_stark`` as Stark proxy).
One-zone: ``I = S (1 - e^{-tau})``; two-zone: outer -> inner -> outer slab
recurrence (:func:`plasma_physics.two_zone_transfer`).  The emergent radiance
is accumulated on a fine grid, convolved with the instrument kernel,
interpolated onto the spectrometer axis and unit-normalised.

Approximations (documented on purpose)
--------------------------------------
* Radiative transfer is solved **per element** and the emergent radiances are
  summed: lines of different elements do not absorb each other.  Lines of the
  *same* element are treated jointly (their kappa/eps add before transfer),
  so overlapping/blended lines of one element are handled exactly.
* Every line only exists within ``+-line_window_nm`` of its centre (thin
  lines lose the same Lorentzian-wing fraction, which cancels in a Boltzmann
  plot).  With ``adaptive_window`` (default on) saturated lines get up to
  8x that window so their black cores are not cut by a box edge; beyond
  that the wings are truncated.
* Only ionisation stages I and II exist in the DB (no stage III at 20 kK).
* Stark widths are not in the DB: one Lorentzian HWHM per zone for all lines.
* Continuum (bremsstrahlung/recombination) is not modelled unless the
  ``augment.continuum`` knob adds a flat pedestal.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

from data import plasma_physics as pp
from data.atomic_data import atomic_mass, mass_to_number_fractions
from data.libs_pipeline import (
    ZONE_COLUMNS,
    SyntheticLIBSDataset,
    _db_elements,
    _get_eion,
    _LEGACY_META_COLS,
    _load_partf,
    _reset_db_caches,
    line_db_cache_key,
    load_sample_types,
    load_wavelength,
    unit_norm,
)

__all__ = [
    "PHYSICS_VERSION", "DEFAULT_ZONE_CFG", "DEFAULT_INSTRUMENT", "ZoneState",
    "zone_states_from_row", "generate_zone_sample_table", "make_fine_grid",
    "instrument_kernel", "synthesise_fine_grid", "synthesise_spectrum",
    "generate_zone_spectra", "TwoZoneSyntheticDataset",
    "build_two_zone_dataset_from_config",
]

PHYSICS_VERSION = 2

DEFAULT_ZONE_CFG: dict[str, list[float]] = {
    "te1": [8000.0, 20000.0],          # K, uniform
    "ne1": [5.0e16, 3.0e18],           # cm^-3, log-uniform
    "te2_ratio": [0.15, 0.6],          # Te2 = Te1 * U(...)
    "ne2_log_ratio": [-3.0, -1.0],     # Ne2 = Ne1 * 10^U(...)
    "l_inner_cm": [0.003, 0.05],       # log-uniform (calibrated on PURE KFE, scripts/calibrate_generator.py)
    "l_outer_cm": [2.0e-5, 2.0e-3],    # log-uniform, each of the two outer segments
    "gamma_stark_nm": [0.003, 0.03],   # Lorentz HWHM per zone, log-uniform
}
DEFAULT_INSTRUMENT: dict[str, Any] = {"profile": "gaussian", "fwhm_nm": 0.047}
DEFAULT_AUGMENT: dict[str, float] = {"noise_sigma": 0.0, "continuum": 0.0}
DEFAULT_FINE_STEP_NM = 0.002
DEFAULT_LINE_WINDOW_NM = 0.4
DEFAULT_MIN_RELATIVE_INTENSITY = 1e-7
DEFAULT_ADAPTIVE_WINDOW = True
ADAPTIVE_WINDOW_MAX_DOUBLINGS = 3        # window <= 8 x line_window_nm
FINE_GRID_PAD_NM = 1.0
_MAX_PROFILE_BLOCK = 4_000_000   # max (n_lines x n_window) elements evaluated at once

_SKIP_COLS = _LEGACY_META_COLS | set(ZONE_COLUMNS)


# ─────────────────────────────────────────────────────────────────────────────
# Plasma state per zone
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ZoneState:
    """One homogeneous LTE slab."""
    T: float          # K
    Ne: float         # cm^-3
    N: float          # heavy-particle density, cm^-3
    l: float          # path length, cm (per traversal)
    gamma_nm: float   # Lorentzian HWHM (Stark proxy), nm


def zone_states_from_row(row: Mapping[str, Any]) -> tuple[ZoneState, ZoneState | None]:
    """(inner, outer) zone states from a sample-table row; ``outer`` is None
    for one-zone shots (``plasma_model != 'two_zone'`` or ``l_outer <= 0``)."""
    inner = ZoneState(float(row["Te1"]), float(row["Ne1"]), float(row["N1"]),
                      float(row["l_inner"]), float(row["gamma_stark1"]))
    if str(row.get("plasma_model", "one_zone")) == "two_zone" and float(row["l_outer"]) > 0:
        outer = ZoneState(float(row["Te2"]), float(row["Ne2"]), float(row["N2"]),
                          float(row["l_outer"]), float(row["gamma_stark2"]))
    else:
        outer = None
    return inner, outer


# ─────────────────────────────────────────────────────────────────────────────
# Vectorised ionisation fractions (for N from quasi-neutrality)
# ─────────────────────────────────────────────────────────────────────────────
def _partition_functions_vec(element: str, T: np.ndarray, db_path: str) -> tuple[np.ndarray, np.ndarray]:
    """U_I(T), U_II(T) for an array of temperatures (same sums as
    ``libs_pipeline.partition_function_cached``, vectorised over T)."""
    gi_I, Ei_I, gi_II, Ei_II = _load_partf(element, db_path)
    kT = pp.KB_EV * np.asarray(T, dtype=np.float64)[:, None]
    U_I = (gi_I[None, :] * np.exp(-Ei_I[None, :] / kT)).sum(axis=1) if gi_I.size else np.zeros(len(T))
    U_II = (gi_II[None, :] * np.exp(-Ei_II[None, :] / kT)).sum(axis=1) if gi_II.size else np.zeros(len(T))
    return U_I, U_II


def _ionised_fraction_vec(element: str, T: np.ndarray, Ne: np.ndarray, db_path: str) -> np.ndarray:
    """r_II = S10 / (1 + S10) for arrays of (T, Ne)."""
    U_I, U_II = _partition_functions_vec(element, T, db_path)
    E_ion = float(_get_eion(element, db_path))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        S10 = pp.saha_ratio(T, Ne, U_I, U_II, E_ion)
        r_II = S10 / (1.0 + S10)
    r_II = np.where(np.isfinite(r_II), r_II, np.where(U_II > 0, 1.0, 0.0))
    return np.clip(r_II, 0.0, 1.0)


def _number_density_rows(
    elements: Sequence[str],
    number_fractions: np.ndarray,
    T: np.ndarray,
    Ne: np.ndarray,
    db_path: str,
    number_density_max: float | None,
) -> np.ndarray:
    """N per row from quasi-neutrality, N = Ne / sum_e x_e r_II,e(T, Ne)."""
    denom = np.zeros(len(T), dtype=np.float64)
    for j, elem in enumerate(elements):
        x = number_fractions[:, j]
        if not np.any(x > 0):
            continue
        denom += x * _ionised_fraction_vec(elem, T, Ne, db_path)
    with np.errstate(divide="ignore"):
        N = np.where(denom > 0, Ne / denom, np.inf)
    if number_density_max is not None:
        N = np.minimum(N, float(number_density_max))
    if not np.all(np.isfinite(N)):
        raise ValueError("quasi-neutrality gave a non-finite N (no ionised species); "
                         "set generation.number_density_max or a numeric number_density")
    return N


def _saha_equilibrium_ne(
    elements: Sequence[str],
    number_fractions: np.ndarray,
    T: np.ndarray,
    N: np.ndarray,
    db_path: str,
    log10_lo: float = 8.0,
    log10_hi: float = 22.0,
    n_iter: int = 60,
) -> np.ndarray:
    """Electron density per row from Saha equilibrium at fixed (T, N):
    solve Ne = N * sum_e x_e r_II,e(T, Ne) by bisection in log10 Ne
    (the right-hand side decreases monotonically with Ne, so the root is unique)."""
    n = len(T)
    lo = np.full(n, log10_lo, dtype=np.float64)
    hi = np.full(n, log10_hi, dtype=np.float64)
    active = [j for j, e in enumerate(elements) if np.any(number_fractions[:, j] > 0)]

    def rhs(log_ne: np.ndarray) -> np.ndarray:
        ne = 10.0 ** log_ne
        acc = np.zeros(n, dtype=np.float64)
        for j in active:
            acc += number_fractions[:, j] * _ionised_fraction_vec(elements[j], T, ne, db_path)
        return N * acc

    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        f = rhs(mid) - 10.0 ** mid      # positive below the root
        lo = np.where(f > 0, mid, lo)
        hi = np.where(f > 0, hi, mid)
    return 10.0 ** (0.5 * (lo + hi))


# ─────────────────────────────────────────────────────────────────────────────
# Sample table (composition + plasma state per shot)
# ─────────────────────────────────────────────────────────────────────────────
def _log_uniform(rng: np.random.Generator, lo: float, hi: float, n: int) -> np.ndarray:
    return 10.0 ** rng.uniform(np.log10(lo), np.log10(hi), n)


def _normalise_zone_cfg(zone_cfg: Mapping[str, Any] | None) -> dict[str, list[float]]:
    """Defaults + user ranges as float pairs (YAML may hand over strings such
    as ``5.0e16`` that PyYAML does not parse as numbers)."""
    zc = {**DEFAULT_ZONE_CFG, **(zone_cfg or {})}
    out: dict[str, list[float]] = {}
    for k, v in zc.items():
        vals = [float(u) for u in np.atleast_1d(v)]
        if len(vals) != 2 or vals[0] > vals[1]:
            raise ValueError(f"zones.{k} must be a [lo, hi] pair, got {v!r}")
        out[k] = vals
    return out


def generate_zone_sample_table(
    concentration_ranges: dict[str, tuple[float, float]],
    n_samples: int,
    sample_id: str,
    sample_name: str,
    zone_cfg: dict[str, Any],
    rng: np.random.Generator,
    two_zone_fraction: float,
    number_density: float | str = "auto",
    db_path: str | None = None,
    number_density_max: float | None = None,
    outer_density: str = "isobaric",
) -> pd.DataFrame:
    """Draw ``n_samples`` shots of one sample type (contract C1).

    Composition: same draw as ``libs_pipeline.generate_sample_table`` —
    independent uniform per element in (c-u, c+u), row-normalised; element
    columns stay **mass** fractions summing to 1.

    Plasma state per row: ``Te1`` uniform, ``Ne1`` log-uniform,
    ``Te2 = Te1 * U(te2_ratio)``, ``l_inner``/``l_outer`` uniform,
    ``gamma_stark1/2`` log-uniform, and ``plasma_model`` two_zone with
    probability ``two_zone_fraction`` (one_zone rows get
    ``Te2=Te1, Ne2=Ne1, l_outer=0, N2=N1``).

    ``N1`` comes from quasi-neutrality at the inner state
    (``number_density="auto"``, needs ``db_path``) or is the given float.
    Outer zone (``outer_density``):
      * ``"isobaric"`` (default): pressure balance ``N2 = N1 * Te1 / Te2`` and
        ``Ne2`` from Saha equilibrium at ``(Te2, N2)`` — the cold shell is
        weakly ionised, as it should be; ``ne2_log_ratio`` is ignored.
      * ``"quasi_neutral"``: legacy rule ``Ne2 = Ne1 * 10^U(ne2_log_ratio)`` and
        ``N2 = Ne2 / sum x r_II`` (blows up for cold shells; kept for ablations).
    ``Te``/``Ne`` alias ``Te1``/``Ne1`` for legacy consumers.
    """
    zc = _normalise_zone_cfg(zone_cfg)
    data: dict[str, Any] = {
        "sample_type_id": [sample_id] * n_samples,
        "sample_type_name": [sample_name] * n_samples,
        "unique_id": [f"{sample_id}_{i+1:04d}" for i in range(n_samples)],
    }
    elements = list(concentration_ranges.keys())
    mat = np.column_stack([
        rng.uniform(lo, hi, n_samples) for lo, hi in concentration_ranges.values()
    ])
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    mat = mat / row_sums
    for j, e in enumerate(elements):
        data[e] = mat[:, j]

    te1 = rng.uniform(zc["te1"][0], zc["te1"][1], n_samples)
    ne1 = _log_uniform(rng, zc["ne1"][0], zc["ne1"][1], n_samples)
    te2 = te1 * rng.uniform(zc["te2_ratio"][0], zc["te2_ratio"][1], n_samples)
    ne2 = ne1 * 10.0 ** rng.uniform(zc["ne2_log_ratio"][0], zc["ne2_log_ratio"][1], n_samples)
    # path lengths span more than a decade (calibration: core N*l 5e15-3e16 cm^-2,
    # shell 1e14-3e14 cm^-2 for PURE KFE) -> log-uniform draws
    l_inner = _log_uniform(rng, zc["l_inner_cm"][0], zc["l_inner_cm"][1], n_samples)
    l_outer = _log_uniform(rng, zc["l_outer_cm"][0], zc["l_outer_cm"][1], n_samples)
    g1 = _log_uniform(rng, zc["gamma_stark_nm"][0], zc["gamma_stark_nm"][1], n_samples)
    g2 = _log_uniform(rng, zc["gamma_stark_nm"][0], zc["gamma_stark_nm"][1], n_samples)
    is_two = rng.random(n_samples) < float(two_zone_fraction)

    if outer_density not in ("isobaric", "quasi_neutral"):
        raise ValueError(f"outer_density must be 'isobaric' or 'quasi_neutral', got {outer_density!r}")
    x = mass_to_number_fractions(mat, elements)
    if isinstance(number_density, str):
        if number_density != "auto":
            raise ValueError(f"number_density must be 'auto' or a float, got {number_density!r}")
        if db_path is None:
            raise ValueError("number_density='auto' requires db_path")
        N1 = _number_density_rows(elements, x, te1, ne1, db_path, number_density_max)
    else:
        N1 = np.full(n_samples, float(number_density))
    N2 = N1.copy()
    if np.any(is_two):
        if outer_density == "isobaric":
            N2[is_two] = N1[is_two] * te1[is_two] / te2[is_two]
            if number_density_max is not None:
                N2[is_two] = np.minimum(N2[is_two], float(number_density_max))
            if db_path is None:
                raise ValueError("outer_density='isobaric' requires db_path (Saha equilibrium for Ne2)")
            ne2[is_two] = _saha_equilibrium_ne(elements, x[is_two], te2[is_two], N2[is_two], db_path)
        else:
            if isinstance(number_density, str):
                N2[is_two] = _number_density_rows(
                    elements, x[is_two], te2[is_two], ne2[is_two], db_path, number_density_max,
                )

    # one-zone rows: collapse zone 2 onto zone 1
    one = ~is_two
    te2[one] = te1[one]
    ne2[one] = ne1[one]
    l_outer[one] = 0.0
    g2[one] = g1[one]
    N2[one] = N1[one]

    data["Te"] = te1
    data["Ne"] = ne1
    data["plasma_model"] = np.where(is_two, "two_zone", "one_zone").astype(object)
    data["Te1"], data["Ne1"], data["Te2"], data["Ne2"] = te1, ne1, te2, ne2
    data["l_inner"], data["l_outer"] = l_inner, l_outer
    data["N1"], data["N2"] = N1, N2
    data["gamma_stark1"], data["gamma_stark2"] = g1, g2
    return pd.DataFrame(data)


# ─────────────────────────────────────────────────────────────────────────────
# Fine grid, profiles, instrument
# ─────────────────────────────────────────────────────────────────────────────
def make_fine_grid(wavelength: np.ndarray, fine_step_nm: float, pad_nm: float = FINE_GRID_PAD_NM) -> np.ndarray:
    """Uniform grid covering the spectrometer axis ``+- pad_nm``."""
    lo = float(np.min(wavelength)) - pad_nm
    hi = float(np.max(wavelength)) + pad_nm
    n = int(np.ceil((hi - lo) / fine_step_nm)) + 1
    return lo + fine_step_nm * np.arange(n, dtype=np.float64)


def instrument_kernel(fine_step_nm: float, instrument: Mapping[str, Any] | None) -> np.ndarray | None:
    """Area-normalised instrument response sampled on the fine grid.

    ``instrument = {profile: gaussian|voigt, fwhm_nm: float[, lorentz_fwhm_nm: float]}``.
    For ``voigt`` the Gaussian part is chosen so the Olivero-Longbothum FWHM
    equals ``fwhm_nm``.  Returns None when no broadening is requested.
    """
    if not instrument:
        return None
    fwhm = float(instrument.get("fwhm_nm", 0.0))
    if fwhm <= 0:
        return None
    profile = str(instrument.get("profile", "gaussian")).lower()
    if profile == "gaussian":
        sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        gamma = 0.0
        half = 5.0 * sigma
    elif profile == "voigt":
        fl = float(instrument.get("lorentz_fwhm_nm", 0.5 * fwhm))
        if fl >= fwhm:
            raise ValueError("instrument.lorentz_fwhm_nm must be smaller than fwhm_nm")
        fg2 = (fwhm - 0.5346 * fl) ** 2 - 0.2166 * fl ** 2
        if fg2 <= 0:
            raise ValueError("instrument.lorentz_fwhm_nm too large for the requested fwhm_nm")
        sigma = np.sqrt(fg2) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        gamma = 0.5 * fl
        half = max(5.0 * sigma, 20.0 * fwhm)
    else:
        raise ValueError(f"unknown instrument profile {profile!r} (gaussian|voigt)")
    n_half = max(1, int(np.ceil(half / fine_step_nm)))
    dl = fine_step_nm * np.arange(-n_half, n_half + 1, dtype=np.float64)
    k = pp.voigt_profile(dl, sigma, gamma)
    return k / k.sum()


def _line_half_widths(
    tau0: np.ndarray, gamma_nm: float, line_window_nm: float, adaptive: bool,
) -> np.ndarray:
    """Per-line half window [nm]. Thin lines use ``line_window_nm``; when
    ``adaptive`` is set, saturated lines get the window doubled (at most
    ``ADAPTIVE_WINDOW_MAX_DOUBLINGS`` times) until the Lorentzian wing optical
    depth tau0 (gamma/dl)^2 at the window edge drops below ~0.01, so that the
    black core of a thick line is not truncated by a box edge."""
    n = tau0.size
    if not adaptive or n == 0:
        return np.full(n, float(line_window_nm))
    need = 10.0 * float(gamma_nm) * np.sqrt(np.maximum(tau0, 1.0))
    k = np.ceil(np.log2(np.maximum(need / line_window_nm, 1.0)))
    k = np.clip(k, 0, ADAPTIVE_WINDOW_MAX_DOUBLINGS)
    return float(line_window_nm) * 2.0 ** k


def _accumulate_line_profiles(
    grid: np.ndarray,
    fine_step_nm: float,
    wl_nm: np.ndarray,
    sigma_nm: np.ndarray,
    gamma_nm: float,
    kappa_int: np.ndarray,
    eps_int: np.ndarray,
    half_width_nm: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """kappa(lambda) [cm^-1] and eps(lambda) [per cm] on the fine grid from
    per-line integrated coefficients, each line spread with its Voigt profile
    inside ``+-half_width_nm`` (searchsorted windows, vectorised over lines;
    lines sharing a half width are processed as one block)."""
    n_grid = grid.size
    kappa = np.zeros(n_grid, dtype=np.float64)
    eps = np.zeros(n_grid, dtype=np.float64)
    n_lines = wl_nm.size
    if n_lines == 0:
        return kappa, eps
    hw_all = np.broadcast_to(np.asarray(half_width_nm, dtype=np.float64), (n_lines,))
    for hw in np.unique(hw_all):
        sel = np.nonzero(hw_all == hw)[0]
        n_w = int(round(2.0 * hw / fine_step_nm)) + 1
        offsets = np.arange(n_w, dtype=np.int64)[None, :]
        block = max(1, _MAX_PROFILE_BLOCK // n_w)
        for s in range(0, sel.size, block):
            ids = sel[s:s + block]
            wl = wl_nm[ids]
            lo = np.searchsorted(grid, wl - hw, side="left")
            idx = lo[:, None] + offsets
            valid = idx < n_grid
            idx = np.minimum(idx, n_grid - 1)
            dl = grid[idx] - wl[:, None]
            phi = pp.voigt_profile(dl, sigma_nm[ids, None], gamma_nm)   # per nm
            phi = np.where(valid, phi, 0.0) / pp.NM_TO_CM                 # per cm
            flat = idx.ravel()
            kappa += np.bincount(flat, weights=(kappa_int[ids, None] * phi).ravel(), minlength=n_grid)
            eps += np.bincount(flat, weights=(eps_int[ids, None] * phi).ravel(), minlength=n_grid)
    return kappa, eps


# ─────────────────────────────────────────────────────────────────────────────
# Forward model
# ─────────────────────────────────────────────────────────────────────────────
def _element_line_sets(
    element: str, inner: ZoneState, outer: ZoneState | None, db_path: str, grid: np.ndarray,
) -> tuple[pp.LineSet, pp.LineSet | None, np.ndarray]:
    """Line sets per zone plus the mask of lines inside the fine grid."""
    ls1 = pp.line_set_for_element(element, inner.T, inner.Ne, db_path)
    ls2 = pp.line_set_for_element(element, outer.T, outer.Ne, db_path) if outer is not None else None
    in_grid = (ls1.wl_nm >= grid[0]) & (ls1.wl_nm <= grid[-1])
    return ls1, ls2, in_grid


def synthesise_fine_grid(
    elements: Sequence[str],
    number_fractions: np.ndarray,
    grid: np.ndarray,
    inner: ZoneState,
    outer: ZoneState | None,
    db_path: str,
    line_window_nm: float = DEFAULT_LINE_WINDOW_NM,
    min_relative_intensity: float = DEFAULT_MIN_RELATIVE_INTENSITY,
    adaptive_window: bool = DEFAULT_ADAPTIVE_WINDOW,
    return_lines: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[dict[str, Any]]]:
    """Emergent spectral radiance on the fine grid [erg s^-1 cm^-2 sr^-1 cm^-1]
    (before instrument broadening, no normalisation).

    ``number_fractions`` are number (mole) fractions over ``elements``.  Lines
    whose optically thin radiance (summed over the slabs along the line of
    sight) is below ``min_relative_intensity`` times the strongest line of the
    spectrum are dropped.  Each line lives inside ``+-line_window_nm``; with
    ``adaptive_window`` saturated lines get a wider window (see
    :func:`_line_half_widths`).  With ``return_lines=True`` a per-element list
    of dicts with the kept lines' ``wl, is_I, Ei, Ek, gi, gk, Ak, I_thin,
    tau0`` (inner-zone line-centre optical depth) is returned as well.
    """
    grid = np.asarray(grid, dtype=np.float64)
    step = float(grid[1] - grid[0])
    x = np.asarray(number_fractions, dtype=np.float64)

    # pass 1: line sets and thin radiances (threshold is global over the spectrum)
    prepared = []
    i_max = 0.0
    for j, elem in enumerate(elements):
        if x[j] <= 0:
            continue
        ls1, ls2, in_grid = _element_line_sets(elem, inner, outer, db_path, grid)
        if len(ls1) == 0 or not np.any(in_grid):
            continue
        n1 = x[j] * inner.N
        i_thin = ls1.emissivity_int(n1) * inner.l
        if outer is not None:
            i_thin = i_thin + 2.0 * ls2.emissivity_int(x[j] * outer.N) * outer.l
        i_thin = np.where(in_grid, i_thin, 0.0)
        i_max = max(i_max, float(i_thin.max()))
        prepared.append((j, elem, ls1, ls2, in_grid, i_thin))

    radiance = np.zeros(grid.size, dtype=np.float64)
    lines_out: list[dict[str, Any]] = []
    if not prepared or i_max <= 0:
        return (radiance, lines_out) if return_lines else radiance
    threshold = float(min_relative_intensity) * i_max

    # pass 2: per-element transfer, summed over elements
    for j, elem, ls1, ls2, in_grid, i_thin in prepared:
        keep = in_grid & (i_thin >= threshold)
        if not np.any(keep):
            continue
        mass = atomic_mass(elem)
        wl = ls1.wl_nm[keep]
        n1 = x[j] * inner.N
        k1_int = ls1.kappa_int(n1)[keep]
        e1_int = ls1.emissivity_int(n1)[keep]
        sig1 = pp.doppler_sigma_nm(wl, inner.T, mass)
        tau0_1 = pp.line_centre_optical_depth(k1_int, sig1, inner.gamma_nm, inner.l)
        hw1 = _line_half_widths(tau0_1, inner.gamma_nm, line_window_nm, adaptive_window)
        kappa1, eps1 = _accumulate_line_profiles(grid, step, wl, sig1, inner.gamma_nm, k1_int, e1_int, hw1)
        with np.errstate(divide="ignore", invalid="ignore"):
            S1 = np.where(kappa1 > 0, eps1 / kappa1, 0.0)
        tau1 = kappa1 * inner.l
        if outer is None:
            radiance += pp.one_zone_transfer(S1, tau1)
        else:
            n2 = x[j] * outer.N
            k2_int = ls2.kappa_int(n2)[keep]
            e2_int = ls2.emissivity_int(n2)[keep]
            sig2 = pp.doppler_sigma_nm(wl, outer.T, mass)
            tau0_2 = pp.line_centre_optical_depth(k2_int, sig2, outer.gamma_nm, outer.l)
            hw2 = _line_half_widths(tau0_2, outer.gamma_nm, line_window_nm, adaptive_window)
            kappa2, eps2 = _accumulate_line_profiles(grid, step, wl, sig2, outer.gamma_nm, k2_int, e2_int, hw2)
            with np.errstate(divide="ignore", invalid="ignore"):
                S2 = np.where(kappa2 > 0, eps2 / kappa2, 0.0)
            tau2 = kappa2 * outer.l
            radiance += pp.two_zone_transfer(S2, tau2, S1, tau1)
        if return_lines:
            lines_out.append({
                "element": elem,
                "wl": wl,
                "is_I": ls1.is_I[keep],
                "Ei": ls1.Ei[keep], "Ek": ls1.Ek[keep],
                "gi": ls1.gi[keep], "gk": ls1.gk[keep], "Ak": ls1.Ak[keep],
                "I_thin": i_thin[keep],
                "tau0": tau0_1,
            })
    return (radiance, lines_out) if return_lines else radiance


def apply_instrument(radiance: np.ndarray, kernel: np.ndarray | None) -> np.ndarray:
    """Convolve the fine-grid radiance with the instrument kernel (same length)."""
    if kernel is None:
        return radiance
    return fftconvolve(radiance, kernel, mode="same")


def synthesise_spectrum(
    elements: Sequence[str],
    mass_fractions: np.ndarray,
    wavelength: np.ndarray,
    row: Mapping[str, Any],
    db_path: str,
    gen_cfg: Mapping[str, Any] | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Unit-normalised synthetic spectrum on the spectrometer axis.

    Args:
        elements:       element symbols (order of ``mass_fractions``)
        mass_fractions: mass fractions (rows of the sample table)
        wavelength:     spectrometer axis [nm] (need not be monotonic)
        row:            sample-table row with the contract-C1 zone columns
        db_path:        SQLite line database
        gen_cfg:        {fine_step_nm, line_window_nm, min_relative_intensity,
                         adaptive_window, instrument: {profile, fwhm_nm},
                         augment: {noise_sigma, continuum}}
        rng:            only used when ``augment.noise_sigma > 0``
    """
    gc = dict(gen_cfg or {})
    fine_step = float(gc.get("fine_step_nm", DEFAULT_FINE_STEP_NM))
    grid = make_fine_grid(wavelength, fine_step)
    inner, outer = zone_states_from_row(row)
    x = mass_to_number_fractions(np.asarray(mass_fractions, dtype=np.float64), elements)

    radiance = synthesise_fine_grid(
        elements, x, grid, inner, outer, db_path,
        line_window_nm=float(gc.get("line_window_nm", DEFAULT_LINE_WINDOW_NM)),
        min_relative_intensity=float(gc.get("min_relative_intensity", DEFAULT_MIN_RELATIVE_INTENSITY)),
        adaptive_window=bool(gc.get("adaptive_window", DEFAULT_ADAPTIVE_WINDOW)),
    )
    radiance = apply_instrument(radiance, instrument_kernel(fine_step, gc.get("instrument", DEFAULT_INSTRUMENT)))
    spectrum = np.interp(np.asarray(wavelength, dtype=np.float64), grid, radiance)

    aug = {**DEFAULT_AUGMENT, **(gc.get("augment") or {})}
    peak = float(spectrum.max()) if spectrum.size else 0.0
    if peak > 0 and float(aug.get("continuum", 0.0)) > 0:
        spectrum = spectrum + float(aug["continuum"]) * peak
    if peak > 0 and float(aug.get("noise_sigma", 0.0)) > 0:
        rng = rng or np.random.default_rng()
        spectrum = spectrum + rng.normal(0.0, float(aug["noise_sigma"]) * peak, spectrum.size)
    return unit_norm(spectrum)


# ─────────────────────────────────────────────────────────────────────────────
# Multiprocessing synthesis over a sample table
# ─────────────────────────────────────────────────────────────────────────────
_zw_wavelength: np.ndarray | None = None
_zw_db_path: str | None = None
_zw_gen_cfg: dict[str, Any] = {}


def _init_zone_worker(wavelength: np.ndarray, db_path: str, gen_cfg: dict[str, Any]) -> None:
    global _zw_wavelength, _zw_db_path, _zw_gen_cfg
    _zw_wavelength = wavelength
    _zw_db_path = db_path
    _zw_gen_cfg = gen_cfg
    _reset_db_caches()


def _generate_zone_one(args) -> tuple[int, np.ndarray]:
    idx, elements, mass_fracs, row = args
    rng = np.random.default_rng(int(_zw_gen_cfg.get("seed", 0)) * 1_000_003 + idx)
    spec = synthesise_spectrum(elements, mass_fracs, _zw_wavelength, row, _zw_db_path, _zw_gen_cfg, rng=rng)
    return idx, spec


def generate_zone_spectra(
    sample_table: pd.DataFrame,
    wavelength: np.ndarray,
    db_path: str,
    gen_cfg: dict[str, Any],
    n_workers: int = 1,
    verbose: bool = True,
) -> np.ndarray:
    """Synthesise every row of ``sample_table`` (contract-C1 columns)."""
    n_samples = len(sample_table)
    # float32: the spectra are unit-normalised and end up in a float32 tensor,
    # and float64 doubles both the 8 GB working set and the HDF5 cache.
    spectra = np.zeros((n_samples, len(wavelength)), dtype=np.float32)
    elements = [c for c in sample_table.columns if c not in _SKIP_COLS]
    conc_mat = sample_table[elements].to_numpy(dtype=np.float64)
    zone_rows = sample_table[list(ZONE_COLUMNS)].to_dict("records")
    tasks = [(i, elements, conc_mat[i], zone_rows[i]) for i in range(n_samples)]

    if n_workers > 1 and n_samples > 1:
        if verbose:
            print(f"   Parallelising across {n_workers} workers...")
        with mp.Pool(
            processes=n_workers,
            initializer=_init_zone_worker,
            initargs=(wavelength, db_path, gen_cfg),
        ) as pool:
            done = 0
            for idx, spec in pool.imap_unordered(_generate_zone_one, tasks, chunksize=4):
                spectra[idx] = spec
                done += 1
                if verbose and done % 200 == 0:
                    print(f"   Completed {done}/{n_samples}")
    else:
        _init_zone_worker(wavelength, db_path, gen_cfg)
        for k, task in enumerate(tasks, start=1):
            idx, spec = _generate_zone_one(task)
            spectra[idx] = spec
            if verbose and k % 200 == 0:
                print(f"   Completed {k}/{n_samples}")

    if verbose:
        print(f"Generated {n_samples} synthetic spectra (physics_version {PHYSICS_VERSION}).")
    return spectra


# ─────────────────────────────────────────────────────────────────────────────
# Dataset wrapper (same HDF5 layout / cache dir as SyntheticLIBSDataset)
# ─────────────────────────────────────────────────────────────────────────────
class TwoZoneSyntheticDataset(SyntheticLIBSDataset):
    """Physics-version-2 synthetic dataset. Same ``synthetic_cache_<key>.h5``
    layout as the legacy class (``spectra`` + ``sample_table`` group, string
    columns stored as variable-length strings); the key covers every knob of
    the new generator plus ``physics_version``."""

    def __init__(
        self,
        sample_types: list[dict[str, Any]],
        wavelength: np.ndarray,
        db_path: str,
        zone_cfg: dict[str, Any] | None = None,
        plasma_model: str = "mixed",
        two_zone_fraction: float = 0.7,
        number_density: float | str = "auto",
        number_density_max: float | None = None,
        outer_density: str = "isobaric",
        instrument: dict[str, Any] | None = None,
        fine_step_nm: float = DEFAULT_FINE_STEP_NM,
        line_window_nm: float = DEFAULT_LINE_WINDOW_NM,
        min_relative_intensity: float = DEFAULT_MIN_RELATIVE_INTENSITY,
        adaptive_window: bool = DEFAULT_ADAPTIVE_WINDOW,
        augment: dict[str, float] | None = None,
        n_workers: int = 1,
        cache_dir: str | None = None,
        seed: int = 42,
        verbose: bool = True,
    ):
        if plasma_model not in ("one_zone", "two_zone", "mixed"):
            raise ValueError(f"plasma_model must be one_zone|two_zone|mixed, got {plasma_model!r}")
        self.zone_cfg = _normalise_zone_cfg(zone_cfg)
        self.plasma_model = plasma_model
        self.two_zone_fraction = {"one_zone": 0.0, "two_zone": 1.0}.get(plasma_model, float(two_zone_fraction))
        self.number_density_setting = number_density if isinstance(number_density, str) else float(number_density)
        self.number_density_max = None if number_density_max is None else float(number_density_max)
        self.outer_density = str(outer_density)
        self.instrument = {**DEFAULT_INSTRUMENT, **(instrument or {})}
        self.instrument["fwhm_nm"] = float(self.instrument["fwhm_nm"])
        self.fine_step_nm = float(fine_step_nm)
        self.line_window_nm = float(line_window_nm)
        self.min_relative_intensity = float(min_relative_intensity)
        self.adaptive_window = bool(adaptive_window)
        self.augment = {k: float(v) for k, v in {**DEFAULT_AUGMENT, **(augment or {})}.items()}
        n_density = float(number_density) if not isinstance(number_density, str) else 0.0
        super().__init__(
            sample_types=sample_types,
            wavelength=wavelength,
            db_path=db_path,
            te_range=tuple(float(v) for v in self.zone_cfg["te1"]),
            ne_range=tuple(float(v) for v in self.zone_cfg["ne1"]),
            n_density=n_density,
            optical_path=float(self.zone_cfg["l_inner_cm"][1]),
            n_workers=n_workers,
            cache_dir=cache_dir,
            seed=seed,
            verbose=verbose,
        )

    @property
    def gen_cfg(self) -> dict[str, Any]:
        """Synthesis knobs handed to the worker processes."""
        return {
            "fine_step_nm": self.fine_step_nm,
            "line_window_nm": self.line_window_nm,
            "min_relative_intensity": self.min_relative_intensity,
            "adaptive_window": self.adaptive_window,
            "instrument": dict(self.instrument),
            "augment": dict(self.augment),
            "seed": self.seed,
        }

    @property
    def cache_key(self) -> str:
        cfg = {
            "physics_version": PHYSICS_VERSION,
            "sample_types": self.sample_types,
            "plasma_model": self.plasma_model,
            "two_zone_fraction": self.two_zone_fraction,
            "zones": {k: [float(v) for v in self.zone_cfg[k]] for k in sorted(self.zone_cfg)},
            "number_density": self.number_density_setting,
            "number_density_max": self.number_density_max,
            "outer_density": self.outer_density,
            "path_length_sampling": "log_uniform",
            "instrument": self.instrument,
            "fine_step_nm": self.fine_step_nm,
            "line_window_nm": self.line_window_nm,
            "min_relative_intensity": self.min_relative_intensity,
            "adaptive_window": self.adaptive_window,
            "augment": self.augment,
            "n_wavelength": int(self.wavelength.size),
            "wavelength_first": float(self.wavelength[0]),
            "wavelength_last": float(self.wavelength[-1]),
            "seed": self.seed,
            **line_db_cache_key(self.db_path),
        }
        return hashlib.md5(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]

    def _build(self) -> tuple[pd.DataFrame, np.ndarray]:
        cache = self._cache_path()
        if os.path.isfile(cache):
            if self.verbose:
                print(f"Loading cached spectra from: {cache}")
            return self._load_cache(cache)

        db_elems = _db_elements(self.db_path)
        tables = []
        for i, st in enumerate(self.sample_types):
            try:
                missing = [e for e in st["concentration_ranges"] if e not in db_elems]
                if missing:
                    if self.verbose:
                        print(f"   WARN: skipping {st['sample_name']} (missing in DB: {missing})")
                    continue
                tables.append(generate_zone_sample_table(
                    concentration_ranges=st["concentration_ranges"],
                    n_samples=st["n_samples"],
                    sample_id=st["sample_id"],
                    sample_name=st["sample_name"],
                    zone_cfg=self.zone_cfg,
                    rng=np.random.default_rng(self.seed + i),
                    two_zone_fraction=self.two_zone_fraction,
                    number_density=self.number_density_setting,
                    db_path=self.db_path,
                    number_density_max=self.number_density_max,
                    outer_density=self.outer_density,
                ))
            except Exception as e:
                if self.verbose:
                    print(f"   WARN: skipping {st.get('sample_name', '?')}: {e}")

        if not tables:
            return pd.DataFrame(), np.empty((0, len(self.wavelength)))

        full = pd.concat(tables, ignore_index=True).fillna(0)
        # keep the contract-C1 column order: meta, elements, aliases, zone columns
        elements = [c for c in full.columns if c not in _SKIP_COLS]
        full = full[["sample_type_id", "sample_type_name", "unique_id"] + elements + ["Te", "Ne"] + list(ZONE_COLUMNS)]
        if self.verbose:
            n_two = int((full["plasma_model"] == "two_zone").sum())
            print(f"\nTotal shots to synthesise: {len(full)}  (two_zone: {n_two}, one_zone: {len(full) - n_two})")

        spectra = generate_zone_spectra(
            sample_table=full,
            wavelength=self.wavelength,
            db_path=self.db_path,
            gen_cfg=self.gen_cfg,
            n_workers=self.n_workers,
            verbose=self.verbose,
        )

        self._save_cache(full, spectra, cache)
        if self.verbose:
            print(f"Cached to: {cache}")
        return full, spectra


def build_two_zone_dataset_from_config(cfg: dict) -> TwoZoneSyntheticDataset:
    """Entry point for ``generation.plasma_model in (one_zone, two_zone, mixed)``
    (dispatched from ``libs_pipeline.build_dataset_from_config``)."""
    paths = cfg["paths"]
    gen = cfg.get("generation", {})

    db_path = str(Path(paths["db"]).expanduser().resolve())
    xlsx_path = str(Path(paths["sample_matrix"]).expanduser().resolve())
    wl_json = str(Path(paths["wavelength_json"]).expanduser().resolve())
    cache_dir = str(Path(paths.get("cache_dir", "external_data/cache")).expanduser().resolve())

    wavelength = load_wavelength(wl_json)
    sample_types = load_sample_types(xlsx_path, db_path)

    n_per_type = gen.get("n_samples_per_type", 1)
    if n_per_type != 1:
        for st in sample_types:
            st["n_samples"] = n_per_type
    max_types = gen.get("max_sample_types")
    if max_types is not None:
        sample_types = sample_types[:max_types]

    zones = dict(gen.get("zones") or {})
    return TwoZoneSyntheticDataset(
        sample_types=sample_types,
        wavelength=wavelength,
        db_path=db_path,
        zone_cfg=zones,
        plasma_model=gen.get("plasma_model", "mixed"),
        two_zone_fraction=float(gen.get("two_zone_fraction", 0.7)),
        number_density=gen.get("number_density", "auto"),
        number_density_max=gen.get("number_density_max"),
        outer_density=str(gen.get("outer_density", "isobaric")),
        instrument=gen.get("instrument"),
        fine_step_nm=float(gen.get("fine_step_nm", DEFAULT_FINE_STEP_NM)),
        line_window_nm=float(gen.get("line_window_nm", DEFAULT_LINE_WINDOW_NM)),
        min_relative_intensity=float(gen.get("min_relative_intensity", DEFAULT_MIN_RELATIVE_INTENSITY)),
        adaptive_window=bool(gen.get("adaptive_window", DEFAULT_ADAPTIVE_WINDOW)),
        augment=gen.get("augment"),
        n_workers=int(gen.get("n_workers", 1)),
        cache_dir=cache_dir,
        seed=int(gen.get("seed", 42)),
        verbose=bool(gen.get("verbose", True)),
    )
