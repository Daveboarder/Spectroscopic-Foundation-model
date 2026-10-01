"""
LTE plasma emission physics — single source of truth for the forward model,
the line dictionary and the calibration-free (CF) solver.

Everything here is numpy, CGS, and Kirchhoff-consistent:

    emissivity / absorption coefficient == Planck function B_lambda(T)

so that the optically thin limit of a line is the textbook

    I_int = (h c / (4 pi lambda)) * A_ki * g_k * exp(-E_k / kT) * n_s / U_s(T) * l

and the optically thick limit saturates at B_lambda(T).  This replaces the
inherited ``Lp = 8 pi h c / (10 lambda^3) * N * exp(-dE/kT) * g_k / g_i``
source term of ``data/libs_pipeline.create_spectra`` and
``external_data/Context/TwoZoneSpectraGenerator.py``, whose thin limit was
proportional to ``lambda * g_k^2 A_k / g_i`` (see plan / ARCHITECTURE notes).

Unit conventions
----------------
* wavelengths are passed in **nm** (as stored in the DB and on the
  spectrometer axis) and converted to cm internally;
* energies (E_i, E_k, E_ion) in **eV**;
* temperatures in K, densities in cm^-3, path lengths in cm;
* line profiles ``phi`` are area-normalised **per nm** (``int phi dlambda_nm = 1``);
  optical depth uses ``phi_cm = phi_nm * 1e7``.

Integrated ("_int") quantities are integrated over wavelength (cm), so

    kappa_int [dimensionless]   = kt [cm^3] * n_s [cm^-3]
    kappa(lambda) [cm^-1]       = kappa_int * phi_cm(lambda)
    tau(lambda)                 = kappa(lambda) * l
    epsilon_int [erg s^-1 cm^-3 sr^-1] = emissivity integrated over the line
    B_lambda [erg s^-1 cm^-2 sr^-1 cm^-1]

Only ionisation stages I and II exist in the database; the Saha split is a
two-stage partition of one element population.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import scipy.constants as const
from scipy.special import wofz

# Constants shared with the legacy generator so Saha factors are bit-identical.
from data.libs_pipeline import _EV_TO_ERG, _H, _KB, _ME, partition_function_cached, _get_eion, _get_quant_param
from data.atomic_data import AMU_G, atomic_mass

C_CGS = const.c * 1e2                     # cm/s  (legacy code used m/s)
KB_EV = 8.617333262e-5                    # eV/K
NM_TO_CM = 1e-7
HC_EV_NM = const.h * const.c / const.e * 1e9   # 1239.84 eV*nm

__all__ = [
    "C_CGS", "KB_EV", "NM_TO_CM", "HC_EV_NM",
    "saha_thermal_factor", "saha_ratio", "stage_fractions",
    "line_absorption_int", "line_emissivity_int", "line_source_function",
    "planck_lambda", "doppler_sigma_nm", "voigt_profile", "voigt_peak",
    "voigt_fwhm", "balmer_stark_hwhm_nm", "line_centre_optical_depth", "curve_of_growth_factor",
    "number_density_from_ne", "one_zone_transfer", "two_zone_transfer",
    "LineSet", "line_set_for_element", "thin_line_intensities",
]


# ─────────────────────────────────────────────────────────────────────────────
# Saha–Boltzmann
# ─────────────────────────────────────────────────────────────────────────────
def saha_thermal_factor(T):
    """2 (2 pi m_e k T / h^2)^{3/2}  [cm^-3]."""
    T = np.asarray(T, dtype=np.float64)
    return 2.0 * ((2.0 * np.pi * _ME * _KB * T) / (_H ** 2)) ** 1.5


def saha_ratio(T, Ne, U_I, U_II, E_ion_eV):
    """S10 = n_II / n_I for one element (identical algebra to the legacy code)."""
    T = np.asarray(T, dtype=np.float64)
    return (
        (U_II / (np.asarray(Ne, dtype=np.float64) * U_I))
        * saha_thermal_factor(T)
        * np.exp(-(E_ion_eV * _EV_TO_ERG) / (_KB * T))
    )


def stage_fractions(S10):
    """(r_I, r_II): fraction of the element population in each stage."""
    S10 = np.asarray(S10, dtype=np.float64)
    return 1.0 / (1.0 + S10), S10 / (1.0 + S10)


# ─────────────────────────────────────────────────────────────────────────────
# Line coefficients (per line, vectorised over lines)
# ─────────────────────────────────────────────────────────────────────────────
def line_absorption_int(wl_nm, Ak, gk, Ei_eV, Ek_eV, T, U):
    """Integrated absorption coefficient per unit species density, ``kt`` [cm^3].

    kappa_int = kt * n_s  where
    kt = lambda^4 / (8 pi c) * A_ki * g_k * exp(-E_i/kT) * (1 - exp(-dE/kT)) / U_s.
    The stimulated-emission factor uses dE = E_k - E_i (DB level energies).
    """
    wl_cm = np.asarray(wl_nm, dtype=np.float64) * NM_TO_CM
    kbT = _KB * np.asarray(T, dtype=np.float64)
    dE = (np.asarray(Ek_eV) - np.asarray(Ei_eV)) * _EV_TO_ERG
    return (
        (wl_cm ** 4 / (8.0 * np.pi * C_CGS))
        * np.asarray(Ak, dtype=np.float64) * np.asarray(gk, dtype=np.float64)
        * np.exp(-np.asarray(Ei_eV) * _EV_TO_ERG / kbT)
        * (1.0 - np.exp(-dE / kbT))
        / U
    )


def line_emissivity_int(wl_nm, Ak, gk, Ek_eV, T, U, n_s):
    """Integrated line emissivity [erg s^-1 cm^-3 sr^-1]:
    (h c / (4 pi lambda)) * A_ki * g_k * exp(-E_k/kT) * n_s / U_s."""
    wl_cm = np.asarray(wl_nm, dtype=np.float64) * NM_TO_CM
    kbT = _KB * np.asarray(T, dtype=np.float64)
    return (
        (_H * C_CGS / (4.0 * np.pi * wl_cm))
        * np.asarray(Ak, dtype=np.float64) * np.asarray(gk, dtype=np.float64)
        * np.exp(-np.asarray(Ek_eV) * _EV_TO_ERG / kbT)
        * n_s / U
    )


def line_source_function(wl_nm, Ei_eV, Ek_eV, T):
    """Planck source function of a line, B_lambda(T) with dE = E_k - E_i:
    2 h c^2 / lambda^5 / (exp(dE/kT) - 1)  [erg s^-1 cm^-2 sr^-1 cm^-1].

    Using the level difference (not h c / lambda) keeps
    emissivity / absorption == source exactly (Kirchhoff) despite rounding
    between DB wavelengths and level energies.
    """
    wl_cm = np.asarray(wl_nm, dtype=np.float64) * NM_TO_CM
    kbT = _KB * np.asarray(T, dtype=np.float64)
    dE = (np.asarray(Ek_eV) - np.asarray(Ei_eV)) * _EV_TO_ERG
    return (2.0 * _H * C_CGS ** 2 / wl_cm ** 5) / np.expm1(dE / kbT)


def planck_lambda(wl_nm, T):
    """Black-body spectral radiance B_lambda(T) [erg s^-1 cm^-2 sr^-1 cm^-1]
    (continuum form, dE = h c / lambda)."""
    wl_cm = np.asarray(wl_nm, dtype=np.float64) * NM_TO_CM
    kbT = _KB * np.asarray(T, dtype=np.float64)
    return (2.0 * _H * C_CGS ** 2 / wl_cm ** 5) / np.expm1(_H * C_CGS / (wl_cm * kbT))


# ─────────────────────────────────────────────────────────────────────────────
# Line profiles (per nm, area-normalised)
# ─────────────────────────────────────────────────────────────────────────────
def doppler_sigma_nm(wl_nm, T, mass_amu):
    """Gaussian sigma of thermal Doppler broadening [nm]:
    lambda * sqrt(k T / (m c^2))."""
    wl_nm = np.asarray(wl_nm, dtype=np.float64)
    return wl_nm * np.sqrt(_KB * np.asarray(T, dtype=np.float64)
                           / (np.asarray(mass_amu, dtype=np.float64) * AMU_G * C_CGS ** 2))


def voigt_profile(dl_nm, sigma_nm, gamma_nm):
    """Area-normalised Voigt profile per nm at offset ``dl_nm`` from the centre.
    ``sigma_nm`` (Gaussian sigma) and ``gamma_nm`` (Lorentzian HWHM) may be
    scalars or arrays broadcastable against ``dl_nm``."""
    sigma_nm = np.maximum(np.asarray(sigma_nm, dtype=np.float64), 1e-9)
    z = (np.asarray(dl_nm, dtype=np.float64) + 1j * np.asarray(gamma_nm, dtype=np.float64)) / (sigma_nm * np.sqrt(2.0))
    return wofz(z).real / (sigma_nm * np.sqrt(2.0 * np.pi))


def voigt_peak(sigma_nm, gamma_nm):
    """Peak value (per nm) of the area-normalised Voigt profile."""
    return voigt_profile(0.0, sigma_nm, gamma_nm)


def voigt_fwhm(sigma_nm, gamma_nm):
    """Olivero–Longbothum FWHM [nm] (same approximation as data/line_features.py)."""
    fl = 2.0 * np.asarray(gamma_nm, dtype=np.float64)
    fg = 2.0 * np.asarray(sigma_nm, dtype=np.float64) * np.sqrt(2.0 * np.log(2.0))
    return 0.5346 * fl + np.sqrt(0.2166 * fl ** 2 + fg ** 2)


# Gigosos, González & Cardeñoso, Spectrochim. Acta B 58 (2003) 1489 (computer-simulated
# Stark profiles, ion dynamics included): Balmer FWHM = a * (Ne / 1e17 cm^-3)^b nm,
# nearly independent of T around 1e4 K.  n_upper: (a [nm], b).
BALMER_STARK_FWHM = {3: (1.098, 0.67903), 4: (4.800, 0.68116)}
H_IONISATION_EV = 13.598
BALMER_LOWER_EV = 10.199  # E(n = 2) of H I


def balmer_stark_hwhm_nm(Ek_eV, Ne):
    """Lorentzian HWHM [nm] of H I Balmer lines (lower level n = 2) at electron density
    ``Ne`` [cm^-3]: half the Gigosos et al. (2003) FWHM.  The upper level is taken from
    ``Ek_eV`` (H alpha: n = 3, H beta: n = 4); lines with n >= 5 use the H beta fit,
    an underestimate for H gamma and higher (weak lines here)."""
    Ek = np.asarray(Ek_eV, dtype=np.float64)
    n_up = np.rint(1.0 / np.sqrt(np.clip(1.0 - Ek / H_IONISATION_EV, 1e-6, None)))
    a = np.where(n_up <= 3, BALMER_STARK_FWHM[3][0], BALMER_STARK_FWHM[4][0])
    b = np.where(n_up <= 3, BALMER_STARK_FWHM[3][1], BALMER_STARK_FWHM[4][1])
    return 0.5 * a * (float(Ne) / 1e17) ** b


def line_centre_optical_depth(kappa_int, sigma_nm, gamma_nm, l_cm):
    """tau_0 = kappa_int * phi_peak[per cm] * l."""
    return np.asarray(kappa_int, dtype=np.float64) * voigt_peak(sigma_nm, gamma_nm) / NM_TO_CM * l_cm


def curve_of_growth_factor(tau0, sigma_nm, gamma_nm, half_width_factor: float = 40.0, n_grid: int = 1025):
    """Self-absorption factor f(tau0) in [0, 1] for a homogeneous slab:

        f = int (1 - exp(-tau0 * phi_hat)) dlambda / (tau0 * int phi_hat dlambda)

    with phi_hat = phi / phi_peak, so that emergent area = thin area * f.
    Vectorised over ``tau0`` (any shape); sigma/gamma broadcast against it.
    """
    tau0 = np.asarray(tau0, dtype=np.float64)
    sigma_nm = np.asarray(sigma_nm, dtype=np.float64)
    gamma_nm = np.asarray(gamma_nm, dtype=np.float64)
    width = voigt_fwhm(sigma_nm, gamma_nm)
    # symmetric grid in units of FWHM, dense near the core
    u = np.linspace(-half_width_factor, half_width_factor, n_grid)
    dl = u[..., :] * np.expand_dims(width, -1)                    # [..., n]
    phi = voigt_profile(dl, np.expand_dims(sigma_nm, -1), np.expand_dims(gamma_nm, -1))
    phi_hat = phi / np.expand_dims(voigt_peak(sigma_nm, gamma_nm), -1)
    t = np.expand_dims(tau0, -1)
    num = np.trapezoid(1.0 - np.exp(-t * phi_hat), dl, axis=-1)
    den = np.trapezoid(phi_hat, dl, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(tau0 > 1e-12, num / (tau0 * den), 1.0)
    return np.clip(f, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Number density and radiative transfer
# ─────────────────────────────────────────────────────────────────────────────
def number_density_from_ne(Ne, x, r_II):
    """Quasi-neutrality with single ionisation: N = Ne / sum_e x_e r_II,e.
    ``x`` are number fractions, ``r_II`` the ionised fraction per element."""
    denom = float(np.sum(np.asarray(x, dtype=np.float64) * np.asarray(r_II, dtype=np.float64)))
    if denom <= 0:
        raise ValueError("No ionised species: cannot derive N from Ne (denominator 0).")
    return float(Ne) / denom


def one_zone_transfer(S, tau):
    """Emergent radiance of a homogeneous slab: S (1 - exp(-tau))."""
    return S * (-np.expm1(-tau))


def two_zone_transfer(S_outer, tau_outer, S_inner, tau_inner):
    """Three-slab recurrence (outer -> inner -> outer) along the line of sight,
    ported unchanged from TwoZoneSpectraGenerator._radiative_transfer_two_zone
    but with Planck source functions. ``tau_*`` may be wavelength arrays."""
    i = S_outer * (-np.expm1(-tau_outer))
    i = i * np.exp(-tau_inner) + S_inner * (-np.expm1(-tau_inner))
    i = i * np.exp(-tau_outer) + S_outer * (-np.expm1(-tau_outer))
    return i


# ─────────────────────────────────────────────────────────────────────────────
# Element line sets from the DB
# ─────────────────────────────────────────────────────────────────────────────
class LineSet:
    """Per-line coefficients for one element at one plasma state.

    Attributes (all arrays of length n_lines unless noted):
        element, T, Ne, E_ion, U_I, U_II, S10, r_I, r_II   (scalars)
        wl_nm, is_I (bool), Ei, Ek, gi, gk, Ak
        kt          integrated absorption per unit species density [cm^3]
        eps_per_n   integrated emissivity per unit *element* density
                    [erg s^-1 sr^-1] (already includes the stage fraction)
        S           Planck source function per line [erg s^-1 cm^-2 sr^-1 cm^-1]
    """

    __slots__ = ("element", "T", "Ne", "E_ion", "U_I", "U_II", "S10", "r_I", "r_II",
                 "wl_nm", "is_I", "Ei", "Ek", "gi", "gk", "Ak", "kt", "eps_per_n", "S")

    def __len__(self) -> int:
        return int(self.wl_nm.size)

    def stage_fraction(self) -> np.ndarray:
        return np.where(self.is_I, self.r_I, self.r_II)

    def kappa_int(self, n_element: float) -> np.ndarray:
        """Integrated absorption coefficient for element density ``n_element``."""
        return self.kt * n_element * self.stage_fraction()

    def emissivity_int(self, n_element: float) -> np.ndarray:
        return self.eps_per_n * n_element


def line_set_for_element(element: str, T: float, Ne: float, db_path: str) -> LineSet:
    """All DB lines of ``element`` with coefficients at (T, Ne)."""
    QP = _get_quant_param(element, db_path)
    ls = LineSet()
    ls.element, ls.T, ls.Ne = element, float(T), float(Ne)
    if QP.empty:
        for k in ("wl_nm", "Ei", "Ek", "gi", "gk", "Ak", "kt", "eps_per_n", "S"):
            setattr(ls, k, np.zeros(0, dtype=np.float64))
        ls.is_I = np.zeros(0, dtype=bool)
        ls.E_ion = ls.U_I = ls.U_II = ls.S10 = ls.r_I = ls.r_II = 0.0
        return ls
    ls.E_ion = float(_get_eion(element, db_path))
    ls.U_I, ls.U_II = partition_function_cached(element, T, db_path)
    ls.S10 = float(saha_ratio(T, Ne, ls.U_I, ls.U_II, ls.E_ion))
    ls.r_I, ls.r_II = (float(v) for v in stage_fractions(ls.S10))

    ls.is_I = (QP["ion_state"] == "I").values
    ls.wl_nm = QP["Wavelength"].values.astype(np.float64)
    ls.Ei = QP["Ei"].values.astype(np.float64)
    ls.Ek = QP["Ek"].values.astype(np.float64)
    ls.gi = QP["gi"].values.astype(np.float64)
    ls.gk = QP["gk"].values.astype(np.float64)
    ls.Ak = QP["Ak"].values.astype(np.float64)
    U = np.where(ls.is_I, ls.U_I, ls.U_II)
    frac = np.where(ls.is_I, ls.r_I, ls.r_II)
    ls.kt = line_absorption_int(ls.wl_nm, ls.Ak, ls.gk, ls.Ei, ls.Ek, T, U)
    ls.eps_per_n = line_emissivity_int(ls.wl_nm, ls.Ak, ls.gk, ls.Ek, T, U, frac)
    ls.S = line_source_function(ls.wl_nm, ls.Ei, ls.Ek, T)
    return ls


def thin_line_intensities(
    element: str, T: float, Ne: float, x_e: float, N: float, l: float, db_path: str,
) -> dict[str, np.ndarray]:
    """Optically thin integrated line radiances [erg s^-1 cm^-2 sr^-1] for one
    element with number fraction ``x_e`` in a plasma of total heavy-particle
    density ``N`` [cm^-3] over path ``l`` [cm].  Returns per-line arrays plus
    ``kappa_int`` (for optical-depth estimates) and ``S`` (source function).
    Replacement for the legacy ``line_dictionary.compute_line_intensities_for_plasma``.
    """
    ls = line_set_for_element(element, T, Ne, db_path)
    n_el = float(x_e) * float(N)
    return {
        "wl": ls.wl_nm,
        "ion_state": np.where(ls.is_I, "I", "II"),
        "is_I": ls.is_I,
        "Ei": ls.Ei, "Ek": ls.Ek, "gi": ls.gi, "gk": ls.gk, "Ak": ls.Ak,
        "I_int": ls.emissivity_int(n_el) * float(l),
        "kappa_int": ls.kappa_int(n_el),
        "S": ls.S,
        "S10": ls.S10, "U_I": ls.U_I, "U_II": ls.U_II, "E_ion": ls.E_ion,
    }
