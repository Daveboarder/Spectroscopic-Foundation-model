"""
Consolidated LIBS synthetic spectrum pipeline.

Inherits the physics-based spectrum synthesis from
Daveboarder/Element_Identification (Sample_bootstrap.py, SpectraGenerator.py,
LIBSmethods.py, readData.py, read_sample_types.py), packaged into a single
file so it has zero coupling to the upstream module layout. Resource files
(SQLite DB with QuantParam/E_ion/PartF_var, sample-matrix .xlsx, and a
VASKUT-style wavelength reference JSON) live under `external_data/` by
default and the paths are config-driven.

Public surface used by training code:
    - load_wavelength(path)                 -> np.ndarray
    - load_sample_types(xlsx_path, db_path) -> list[dict]
    - SyntheticLIBSDataset(...)             -> torch.utils.data.Dataset
                                                  attributes: spectra, sample_table

The dataset caches the generated spectra to HDF5 keyed by config hash, so
subsequent runs with the same physical config skip regeneration.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import scipy.constants as const
import torch
from numpy.polynomial import Polynomial
from scipy.special import wofz
from torch.utils.data import Dataset

# ─────────────────────────────────────────────────────────────────────────────
# Physical constants (CGS units, matching upstream SpectraGenerator)
# ─────────────────────────────────────────────────────────────────────────────
_KB = const.k * 1e7              # erg/K
_H = const.h * 1e7               # erg*s
_C_SPEED = const.c               # m/s
_ME = const.electron_mass * 1e3  # g
_EV_TO_ERG = 1.60217e-12

# Default plasma & sample params (overridable per call)
DEFAULT_TE_RANGE = (6500.0, 11000.0)         # K
DEFAULT_NE_RANGE = (1e17, 5e17)              # cm^-3
DEFAULT_OPTICAL_PATH = 1.4e-4                # cm
DEFAULT_NUMBER_DENSITY = 1e-4                # cm^-3 (sample-level)

# Voigt profile widths used for line broadening (upstream calibration)
_GAMMA_FIT = 0.1
_SIGMA_FIT = 0.006
_SIGMA_SQRT2 = _SIGMA_FIT * np.sqrt(2)
_VOIGT_NORM = 1.0 / (_SIGMA_FIT * np.sqrt(2 * np.pi))

# Element-name filters (upstream conventions)
_EXCLUDED_ELEMENTS = {"", "n", "r"}
# LIBS_data.db (air wavelengths) also carries rows under pseudo-elements such as
# "Al-II" / "Mn-II" plus "", "n", "r"; only plain chemical symbols are elements.
_ELEMENT_SYMBOL_RE = re.compile(r"[A-Z][a-z]?")

# Line DB whose cache keys predate the `line_db` key; other DBs (e.g. the air-wavelength
# LIBS_data.db) are added to every spectra cache key so their caches never collide.
LEGACY_LINE_DB = "LIBS_data_vacuum.db"


def is_element_symbol(name: Any) -> bool:
    """True for a plain chemical symbol ("Fe"), False for DB artefacts ("Al-II", "", "n")."""
    s = str(name).strip() if name is not None else ""
    return s not in _EXCLUDED_ELEMENTS and bool(_ELEMENT_SYMBOL_RE.fullmatch(s))


_line_db_md5_cache: dict[tuple[str, int, int], str] = {}


def line_db_cache_key(db_path: str) -> dict[str, str]:
    """Cache-key entry identifying the line DB by name and content (edits to the DB file
    invalidate caches); empty for the legacy vacuum DB so existing hashes stay valid."""
    p = Path(db_path)
    if p.name == LEGACY_LINE_DB:
        return {}
    st = p.stat()
    sig = (str(p.resolve()), st.st_size, st.st_mtime_ns)
    if sig not in _line_db_md5_cache:
        _line_db_md5_cache[sig] = hashlib.md5(p.read_bytes()).hexdigest()[:12]
    return {"line_db": p.name, "line_db_md5": _line_db_md5_cache[sig]}

# Plasma-state columns written by data/two_zone_pipeline.py (physics_version 2).
# They are metadata, never element concentrations. ``Te``/``Ne`` stay as the
# aliases of ``Te1``/``Ne1`` so every legacy consumer keeps working.
ZONE_COLUMNS = (
    "plasma_model", "Te1", "Ne1", "Te2", "Ne2", "l_inner", "l_outer",
    "N1", "N2", "gamma_stark1", "gamma_stark2",
)
_LEGACY_META_COLS = {"sample_type_id", "sample_type_name", "unique_id", "Te", "Ne"}
_PLASMA_MODELS_V2 = ("one_zone", "two_zone", "mixed")
_SPLIT_STRATEGIES = ("random", "group_sample", "group_instrument")


# ─────────────────────────────────────────────────────────────────────────────
# SQLite caches — populated per-process. Multiprocessing workers reset these
# inside _init_worker because forked SQLite connections aren't safe to share.
# ─────────────────────────────────────────────────────────────────────────────
_db_conn: sqlite3.Connection | None = None
_quant_cache: dict[str, pd.DataFrame] = {}
_eion_cache: dict[str, float] = {}
_partf_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}


def _connect(db_path: str) -> sqlite3.Connection:
    global _db_conn
    if _db_conn is None:
        _db_conn = sqlite3.connect(db_path)
    return _db_conn


def _reset_db_caches() -> None:
    """Called inside each worker process after spawn/fork to drop stale state."""
    global _db_conn
    _db_conn = None
    _quant_cache.clear()
    _eion_cache.clear()
    _partf_cache.clear()


def _get_quant_param(element: str, db_path: str) -> pd.DataFrame:
    if element not in _quant_cache:
        cur = _connect(db_path).cursor()
        cur.execute(
            "SELECT Elem_name, ion_state, Wavelength, Ei, Ek, gi, gk, Ak "
            "FROM QuantParam WHERE Elem_name = ?",
            (element,),
        )
        _quant_cache[element] = pd.DataFrame(
            cur.fetchall(),
            columns=["Elem_name", "ion_state", "Wavelength", "Ei", "Ek", "gi", "gk", "Ak"],
        )
    return _quant_cache[element]


def _get_eion(element: str, db_path: str) -> float:
    if element not in _eion_cache:
        cur = _connect(db_path).cursor()
        cur.execute("SELECT Eion FROM E_ion WHERE Elem_name = ?", (element + "+I",))
        row = cur.fetchall()
        if not row:
            raise ValueError(f"E_ion missing for '{element}+I'")
        _eion_cache[element] = row[0][0]
    return _eion_cache[element]


def _load_partf(element: str, db_path: str):
    if element not in _partf_cache:
        cur = _connect(db_path).cursor()
        cur.execute(
            "SELECT ion_state, Ei, gi FROM PartF_var WHERE Elem_name = ?",
            (element,),
        )
        gi_I, Ei_I, gi_II, Ei_II = [], [], [], []
        for ion_state, Ei, gi in cur.fetchall():
            if ion_state == "I":
                gi_I.append(gi); Ei_I.append(Ei)
            elif ion_state == "II":
                gi_II.append(gi); Ei_II.append(Ei)
        if element == "H" and not gi_II:
            # H II is a bare proton: U_II = 1. PartF_var has no level for it, and U_II = 0
            # would make the Saha ratio vanish (hydrogen treated as fully neutral).
            gi_II.append(1.0); Ei_II.append(0.0)
        _partf_cache[element] = (
            np.array(gi_I, dtype=np.float64),
            np.array(Ei_I, dtype=np.float64),
            np.array(gi_II, dtype=np.float64),
            np.array(Ei_II, dtype=np.float64),
        )
    return _partf_cache[element]


def partition_function_cached(element: str, T: float, db_path: str) -> tuple[float, float]:
    """Boltzmann partition functions for neutral (I) and singly ionised (II) species."""
    if T <= 0:
        raise ValueError(f"Temperature must be positive, got T={T} K.")
    gi_I, Ei_I, gi_II, Ei_II = _load_partf(element, db_path)
    kb_eV = 8.617333262e-5  # eV/K
    U_I = float(np.sum(gi_I * np.exp(-Ei_I / (kb_eV * T)))) if gi_I.size else 0.0
    U_II = float(np.sum(gi_II * np.exp(-Ei_II / (kb_eV * T)))) if gi_II.size else 0.0
    return U_I, U_II


# ─────────────────────────────────────────────────────────────────────────────
# Voigt broadening: per-line windowed accumulation (faster than full broadcast)
# ─────────────────────────────────────────────────────────────────────────────
def _voigt_all_lines(
    wavelength: np.ndarray,
    line_centres: np.ndarray,
    amplitudes: np.ndarray,
    window_nm: float = 1.5,
) -> np.ndarray:
    out = np.zeros_like(wavelength)
    sorted_idx = np.argsort(wavelength)
    wl_sorted = wavelength[sorted_idx]
    for k in range(line_centres.size):
        centre, amp = line_centres[k], amplitudes[k]
        lo = np.searchsorted(wl_sorted, centre - window_nm, side="left")
        hi = np.searchsorted(wl_sorted, centre + window_nm, side="right")
        if lo >= hi:
            continue
        idx = sorted_idx[lo:hi]
        z = (wavelength[idx] - centre + 1j * _GAMMA_FIT) / _SIGMA_SQRT2
        out[idx] += amp * wofz(z).real * _VOIGT_NORM
    return out


def create_spectra(
    element: str,
    wavelength: np.ndarray,
    Te: float,
    Ne: float,
    N: float,
    C: float,
    l: float,
    db_path: str,
) -> np.ndarray:
    """Physics-based synthetic emission spectrum for one element. See upstream
    SpectraGenerator.create_spectra for the derivation."""
    QP = _get_quant_param(element, db_path)
    if QP.empty:
        return np.zeros_like(wavelength, dtype=float)

    E_ion = _get_eion(element, db_path)
    PF_I, PF_II = partition_function_cached(element, Te, db_path)

    # Saha ratio S10 = n_II / n_I
    S10 = (
        ((2 * PF_II) / (Ne * PF_I))
        * ((_ME * _KB * Te) / ((_H ** 2) / (2 * np.pi))) ** 1.5
        * np.exp(-(E_ion * _EV_TO_ERG) / (_KB * Te))
    )

    ion_is_I = (QP["ion_state"] == "I").values
    pf_per_line = np.where(ion_is_I, PF_I, PF_II)
    ri = np.where(ion_is_I, 1 / (1 + S10), S10 / (1 + S10))

    wl = QP["Wavelength"].values
    Ak = QP["Ak"].values
    gk = QP["gk"].values
    gi = QP["gi"].values
    Ei = QP["Ei"].values
    Ek = QP["Ek"].values
    kbT = _KB * Te

    kt = (
        (wl ** 4 / (8 * np.pi * _C_SPEED))
        * (Ak * gk * np.exp(-Ei * _EV_TO_ERG / kbT))
        * (1 - np.exp(-_EV_TO_ERG * (Ek - Ei) / kbT))
        / pf_per_line
    )
    Lp = (
        (8 * np.pi * _H * _C_SPEED) / (10 * wl ** 3)
        * N * np.exp(-_EV_TO_ERG * (Ek - Ei) / kbT) * (gk / gi)
    )

    tau = C * N * ri * l * kt
    Ifin = Lp * (1 - np.exp(-tau))

    return _voigt_all_lines(np.asarray(wavelength, dtype=np.float64), wl, Ifin)


# ─────────────────────────────────────────────────────────────────────────────
# Wavelength loading from VASKUT-style spectrometer JSON
# ─────────────────────────────────────────────────────────────────────────────
def compute_ccd_wavelengths(
    n_pixels: int,
    pixel_to_wl: list[float],
    drift: list[float],
) -> np.ndarray:
    """Per-pixel wavelength axis from VASKUT drift + pixel→wavelength polynomials.

    Intensity at pixel index *i* must be paired with ``wp[i]`` — do not replace
    with ``np.linspace(wp.min(), wp.max(), n)`` because the mapping is
    nonlinear and causes multi-nm line-position errors.
    """
    pixels = np.arange(0, n_pixels) + 1
    return Polynomial(pixel_to_wl)(Polynomial(drift)(pixels))


def _get_one_ccd_range(json_path: str, run_id: int, integration_phase: int, ccd_range: int):
    j = json.load(open(json_path))["analysis"]
    for run in j["results"]:
        if run["runId"] == run_id:
            break
    ccd_data = run["spectraData"]
    for phase, cd in enumerate(ccd_data):
        if phase + 1 == integration_phase:
            break
    rng = cd["spectra"][ccd_range - 1]
    I = np.asarray(rng["results"])
    drift = [rng["drift"]["beta"], rng["drift"]["alpha"]]
    p2w = rng["pixelToWaveLength"][::-1]
    # Native spectrometer pixel grid (no upsampling). Upstream uses I.size*4
    # for fitting; for a transformer encoder it just inflates n_bins (and
    # attention cost is O(n^2)).
    return compute_ccd_wavelengths(I.size, p2w, drift)


def load_wavelength(json_path: str, run_id: int = 1, integration_phase: int = 1) -> np.ndarray:
    """Build the full LIBS wavelength axis by concatenating the two CCD ranges
    of a VASKUT-style analysis JSON. Returns shape (N,) — used as n_bins."""
    w1 = _get_one_ccd_range(json_path, run_id, integration_phase, 1)
    w2 = _get_one_ccd_range(json_path, run_id, integration_phase, 2)
    return np.concatenate([w1, w2])


# ─────────────────────────────────────────────────────────────────────────────
# Sample-type loading from Excel matrix (Concentrations + Uncertainties sheets)
# ─────────────────────────────────────────────────────────────────────────────
def _db_elements(db_path: str) -> set[str]:
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT Elem_name FROM QuantParam")
        return {str(r[0]).strip() for r in cur.fetchall() if is_element_symbol(r[0])}


def _normalize_sample_id(name: str, row: int) -> str:
    norm = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").upper()
    return norm if norm else f"SAMPLE_{row:03d}"


def load_sample_types(xlsx_path: str, db_path: str) -> list[dict[str, Any]]:
    """Build sample-type configs from the Excel matrix.

    Each row in the 'Concentrations' sheet becomes one sample type; uncertainties
    in the 'Uncertainties' sheet define ± ranges around the nominal concentration.
    Elements not present in the SQLite DB are silently dropped (matches upstream).
    """
    conc = pd.read_excel(xlsx_path, sheet_name="Concentrations")
    unc = pd.read_excel(xlsx_path, sheet_name="Uncertainties")

    if conc.empty:
        return []

    conc.columns = [str(c).strip() for c in conc.columns]
    unc.columns = [str(c).strip() for c in unc.columns]

    name_col = conc.columns[0]
    elem_cols = [
        c for c in conc.columns[1:]
        if c and not c.lower().startswith("unnamed:")
    ]
    db_elems = _db_elements(db_path)
    elem_cols = [e for e in elem_cols if e in db_elems]

    unc = unc.set_index(unc.columns[0])

    samples: list[dict[str, Any]] = []
    for row_idx, (_, row) in enumerate(conc.iterrows(), start=1):
        raw_name = row.get(name_col, "")
        if pd.isna(raw_name) or str(raw_name).strip() == "":
            continue
        sname = str(raw_name).strip()
        urow = unc.loc[sname] if sname in unc.index else pd.Series(dtype=float)
        if isinstance(urow, pd.DataFrame):
            urow = urow.iloc[0]

        ranges: dict[str, tuple[float, float]] = {}
        for elem in elem_cols:
            val = pd.to_numeric(row.get(elem, 0), errors="coerce")
            c_val = 0.0 if pd.isna(val) else float(val)
            u_val = pd.to_numeric(urow.get(elem, None), errors="coerce")
            u = 0.01 * abs(c_val) if pd.isna(u_val) else float(u_val)
            ranges[elem] = (c_val - u, c_val + u)

        samples.append({
            "sample_id": _normalize_sample_id(sname, row_idx),
            "sample_name": sname,
            "n_samples": 1,           # caller overrides via config
            "concentration_ranges": ranges,
        })
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Sample-table generation (concentrations + plasma params per simulated shot)
# ─────────────────────────────────────────────────────────────────────────────
def generate_sample_table(
    concentration_ranges: dict[str, tuple[float, float]],
    n_samples: int,
    sample_id: str,
    sample_name: str,
    te_range: tuple[float, float],
    ne_range: tuple[float, float],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Draw `n_samples` synthetic shots from this sample type. Concentrations are
    drawn independently per element from their (c-u, c+u) range, then row-
    normalised to sum to 1.0. Te is uniform; Ne is log-uniform."""
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
    data["Te"] = rng.uniform(te_range[0], te_range[1], n_samples)
    log_lo, log_hi = np.log10(ne_range[0]), np.log10(ne_range[1])
    data["Ne"] = 10 ** rng.uniform(log_lo, log_hi, n_samples)
    return pd.DataFrame(data)


def unit_norm(x: np.ndarray) -> np.ndarray:
    """Min-shift then divide by max (upstream convention)."""
    x = x - x.min()
    m = x.max()
    return x if m == 0 else x / m


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample spectrum synthesis — multiprocessing-ready
# ─────────────────────────────────────────────────────────────────────────────
_w_wavelength: np.ndarray | None = None
_w_db_path: str | None = None
_w_n_density: float = 0.0
_w_optical_path: float = 0.0


def _init_worker(wavelength: np.ndarray, db_path: str, n_density: float, optical_path: float):
    global _w_wavelength, _w_db_path, _w_n_density, _w_optical_path
    _w_wavelength = wavelength
    _w_db_path = db_path
    _w_n_density = n_density
    _w_optical_path = optical_path
    _reset_db_caches()


def _generate_one(args) -> tuple[int, np.ndarray]:
    idx, elements, concs, Te, Ne = args
    spec = np.zeros(len(_w_wavelength))
    for j, elem in enumerate(elements):
        if concs[j] > 0:
            try:
                spec += create_spectra(
                    element=elem,
                    wavelength=_w_wavelength,
                    Te=Te, Ne=Ne,
                    N=_w_n_density, C=concs[j], l=_w_optical_path,
                    db_path=_w_db_path,
                )
            except Exception:
                # Bad element / DB miss — leave zero contribution.
                pass
    return idx, unit_norm(spec)


def generate_synthetic_spectra(
    sample_table: pd.DataFrame,
    wavelength: np.ndarray,
    db_path: str,
    n_density: float = DEFAULT_NUMBER_DENSITY,
    optical_path: float = DEFAULT_OPTICAL_PATH,
    n_workers: int = 1,
    verbose: bool = True,
) -> np.ndarray:
    n_samples = len(sample_table)
    spectra = np.zeros((n_samples, len(wavelength)))
    skip = _LEGACY_META_COLS | set(ZONE_COLUMNS)
    elements = [c for c in sample_table.columns if c not in skip]

    Te_arr = sample_table["Te"].values
    Ne_arr = sample_table["Ne"].values
    conc_mat = sample_table[elements].values

    tasks = [
        (i, elements, conc_mat[i], Te_arr[i], Ne_arr[i])
        for i in range(n_samples)
    ]

    if n_workers > 1 and n_samples > 1:
        if verbose:
            print(f"   Parallelising across {n_workers} workers...")
        with mp.Pool(
            processes=n_workers,
            initializer=_init_worker,
            initargs=(wavelength, db_path, n_density, optical_path),
        ) as pool:
            for idx, spec in pool.imap_unordered(_generate_one, tasks, chunksize=4):
                spectra[idx] = spec
                if verbose and (idx + 1) % 200 == 0:
                    print(f"   Completed {idx + 1}/{n_samples}")
    else:
        _init_worker(wavelength, db_path, n_density, optical_path)
        for task in tasks:
            idx, spec = _generate_one(task)
            spectra[idx] = spec
            if verbose and (idx + 1) % 200 == 0:
                print(f"   Completed {idx + 1}/{n_samples}")

    if verbose:
        print(f"Generated {n_samples} synthetic spectra.")
    return spectra


# ─────────────────────────────────────────────────────────────────────────────
# HDF5 cache helpers (shared by synthetic and measured datasets)
# ─────────────────────────────────────────────────────────────────────────────
def save_spectra_cache(table: pd.DataFrame, spectra: np.ndarray, path: str) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset("spectra", data=spectra, compression="gzip")
        grp = f.create_group("sample_table")
        grp.attrs["columns"] = json.dumps(list(table.columns))
        str_dt = h5py.string_dtype()
        for col in table.columns:
            vals = table[col].values
            if vals.dtype.kind in ("U", "O"):
                grp.create_dataset(col, data=list(vals), dtype=str_dt)
            else:
                grp.create_dataset(col, data=vals)


def load_spectra_cache(path: str) -> tuple[pd.DataFrame, np.ndarray]:
    with h5py.File(path, "r") as f:
        # read_direct converts chunk by chunk, so a float64 cache (written
        # before the generator switched to float32) never costs 2x the RAM.
        ds = f["spectra"]
        spectra = np.empty(ds.shape, dtype=np.float32)
        ds.read_direct(spectra)
        grp = f["sample_table"]
        cols = json.loads(grp.attrs["columns"])
        data = {}
        for c in cols:
            v = grp[c][:]
            if v.dtype.kind in ("S", "O"):
                v = [s.decode("utf-8") if isinstance(s, bytes) else s for s in v]
            data[c] = v
    return pd.DataFrame(data), spectra


# ─────────────────────────────────────────────────────────────────────────────
# Dataset wrapper + HDF5 cache
# ─────────────────────────────────────────────────────────────────────────────
class SyntheticLIBSDataset(Dataset):
    """PyTorch dataset that materialises synthetic LIBS spectra in __init__
    (so .spectra / .sample_table are available as numpy arrays for downstream
    train/val splitting). Spectra are cached to HDF5 keyed by config hash."""

    def __init__(
        self,
        sample_types: list[dict[str, Any]],
        wavelength: np.ndarray,
        db_path: str,
        te_range: tuple[float, float] = DEFAULT_TE_RANGE,
        ne_range: tuple[float, float] = DEFAULT_NE_RANGE,
        n_density: float = DEFAULT_NUMBER_DENSITY,
        optical_path: float = DEFAULT_OPTICAL_PATH,
        n_workers: int = 1,
        cache_dir: str | None = None,
        seed: int = 42,
        verbose: bool = True,
    ):
        self.sample_types = sample_types
        self.wavelength = wavelength
        self.db_path = db_path
        self.te_range = te_range
        self.ne_range = ne_range
        self.n_density = n_density
        self.optical_path = optical_path
        self.n_workers = n_workers
        self.seed = seed
        self.verbose = verbose
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), "..", "external_data", "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.sample_table, self.spectra = self._build()

    @property
    def cache_key(self) -> str:
        """12-char md5 fingerprint of the generation config. Used both for the
        spectra HDF5 cache and for the matching splits JSON."""
        cfg = {
            "sample_types": self.sample_types,
            "te_range": list(self.te_range),
            "ne_range": [float(self.ne_range[0]), float(self.ne_range[1])],
            "n_density": self.n_density,
            "optical_path": self.optical_path,
            "n_wavelength": int(self.wavelength.size),
            "wavelength_first": float(self.wavelength[0]),
            "wavelength_last": float(self.wavelength[-1]),
            "seed": self.seed,
            **line_db_cache_key(self.db_path),
        }
        return hashlib.md5(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]

    def _cache_path(self) -> str:
        return os.path.join(self.cache_dir, f"synthetic_cache_{self.cache_key}.h5")

    def _save_cache(self, table: pd.DataFrame, spectra: np.ndarray, path: str):
        save_spectra_cache(table, spectra, path)

    def _load_cache(self, path: str) -> tuple[pd.DataFrame, np.ndarray]:
        return load_spectra_cache(path)

    def _build(self) -> tuple[pd.DataFrame, np.ndarray]:
        cache = self._cache_path()
        if os.path.isfile(cache):
            if self.verbose:
                print(f"Loading cached spectra from: {cache}")
            return self._load_cache(cache)

        rng = np.random.default_rng(self.seed)
        tables = []
        for i, st in enumerate(self.sample_types):
            try:
                # Quick element validation (raise on missing)
                missing = [e for e in st["concentration_ranges"] if e not in _db_elements(self.db_path)]
                if missing:
                    if self.verbose:
                        print(f"   WARN: skipping {st['sample_name']} (missing in DB: {missing})")
                    continue
                tables.append(generate_sample_table(
                    concentration_ranges=st["concentration_ranges"],
                    n_samples=st["n_samples"],
                    sample_id=st["sample_id"],
                    sample_name=st["sample_name"],
                    te_range=self.te_range,
                    ne_range=self.ne_range,
                    rng=np.random.default_rng(self.seed + i),
                ))
            except Exception as e:
                if self.verbose:
                    print(f"   WARN: skipping {st.get('sample_name', '?')}: {e}")

        if not tables:
            return pd.DataFrame(), np.empty((0, len(self.wavelength)))

        full = pd.concat(tables, ignore_index=True).fillna(0)
        if self.verbose:
            print(f"\nTotal shots to synthesise: {len(full)}")

        spectra = generate_synthetic_spectra(
            sample_table=full,
            wavelength=self.wavelength,
            db_path=self.db_path,
            n_density=self.n_density,
            optical_path=self.optical_path,
            n_workers=self.n_workers,
            verbose=self.verbose,
        )

        self._save_cache(full, spectra, cache)
        if self.verbose:
            print(f"Cached to: {cache}")
        return full, spectra

    def __len__(self) -> int:
        return len(self.sample_table)

    def __getitem__(self, idx: int) -> dict:
        row = self.sample_table.iloc[idx].to_dict()
        row["spectrum"] = self.spectra[idx]
        return row


# ─────────────────────────────────────────────────────────────────────────────
# Downstream-label helpers: concentration extraction, splits, clustering
# ─────────────────────────────────────────────────────────────────────────────
_META_COLS = _LEGACY_META_COLS | set(ZONE_COLUMNS)


def extract_finetune_labels(
    sample_table: pd.DataFrame,
    elements: list[str] | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Pull regression targets and origin labels from the generated sample table.

    Args:
        sample_table: as returned by SyntheticLIBSDataset.sample_table
        elements: optional whitelist; default = every non-meta column present

    Returns:
        concentrations: float32 array, shape [N, n_elements], rows sum to ~1
        element_names: list of length n_elements (column order in `concentrations`)
        sample_type_ids: int64 array, shape [N], dense integer encoding of
                         sample_type_id (useful as a fallback class label)
    """
    if elements is None:
        elements = [c for c in sample_table.columns if c not in _META_COLS]
    missing = [e for e in elements if e not in sample_table.columns]
    if missing:
        raise KeyError(f"requested elements not in sample_table: {missing}")

    conc = sample_table[elements].to_numpy(dtype=np.float32, copy=True)
    np.nan_to_num(conc, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    np.clip(conc, 0.0, 1.0, out=conc)

    ids = sample_table["sample_type_id"].astype(str).to_numpy()
    _, inv = np.unique(ids, return_inverse=True)
    return conc, list(elements), inv.astype(np.int64)


def extract_plasma_targets(sample_table: pd.DataFrame) -> dict[str, np.ndarray]:
    """Per-shot plasma-state targets for the calibration-free (CF) task.

    Reads the physics_version-2 columns written by
    :mod:`data.two_zone_pipeline` (``Te1, Ne1, N1, l_inner, plasma_model``).

    Returns float32 arrays of length N:
        Te                inner-zone temperature [K]
        log10_Ne          log10 of the inner-zone electron density [cm^-3]
        log10_Nl          log10(N1 * l_inner) [cm^-2], the column density that
                          sets the optical depth of the inner zone
        is_two_zone       1 for two-zone shots, 0 otherwise
        has_plasma_labels 1 when the row carries valid (positive) plasma
                          labels; 0 (with every other entry zero) when the
                          zone columns are missing or zero, i.e. for measured
                          spectra and for the legacy (physics_version 1) cache.
    """
    n = len(sample_table)
    out = {
        k: np.zeros(n, dtype=np.float32)
        for k in ("Te", "log10_Ne", "log10_Nl", "is_two_zone", "has_plasma_labels")
    }
    needed = ("Te1", "Ne1", "N1", "l_inner")
    if n == 0 or any(c not in sample_table.columns for c in needed):
        return out

    te = pd.to_numeric(sample_table["Te1"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    ne = pd.to_numeric(sample_table["Ne1"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    nd = pd.to_numeric(sample_table["N1"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    li = pd.to_numeric(sample_table["l_inner"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    nl = nd * li
    has = (te > 0) & (ne > 0) & (nl > 0)

    out["Te"][has] = te[has]
    out["log10_Ne"][has] = np.log10(ne[has])
    out["log10_Nl"][has] = np.log10(nl[has])
    if "plasma_model" in sample_table.columns:
        two = sample_table["plasma_model"].astype(str).to_numpy() == "two_zone"
        out["is_two_zone"] = (two & has).astype(np.float32)
    out["has_plasma_labels"] = has.astype(np.float32)
    return out


def measured_groups(sample_table: pd.DataFrame, by: str = "sample") -> np.ndarray:
    """Group ids for grouped train/val/test splits of measured spectra.

    ``by="sample"``      -> ``sample_type_id`` (one group per physical sample)
    ``by="instrument"``  -> instrument token parsed from ``unique_id`` of the
                            form ``{sample_type_id}_{INSTRUMENT}_R{run:02d}``
                            (see data/measured_pipeline.build_measured_entries),
                            e.g. ``PURE_KFE_REMUS_9951601_R02`` -> ``REMUS_9951601``.
                            Rows that do not follow the pattern get ``"unknown"``.
    Returns an array of strings, one per row.
    """
    if by == "sample":
        return sample_table["sample_type_id"].astype(str).to_numpy()
    if by != "instrument":
        raise ValueError(f"measured_groups: unknown grouping {by!r} (use 'sample' or 'instrument')")
    sids = sample_table["sample_type_id"].astype(str).to_numpy()
    uids = sample_table["unique_id"].astype(str).to_numpy()
    pat = re.compile(r"^(?P<inst>.+)_R(?P<run>\d+)$")
    out = []
    for sid, uid in zip(sids, uids):
        rest = uid[len(sid) + 1:] if uid.startswith(sid + "_") else uid
        m = pat.match(rest)
        out.append(m.group("inst") if m else "unknown")
    return np.asarray(out, dtype=str)


def make_splits(
    n: int,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Deterministic random partition. Train + val + test = n; test is held out
    from both pretrain and finetune model selection."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = max(1, int(n * test_fraction)) if test_fraction > 0 else 0
    n_val = max(1, int(n * val_fraction)) if val_fraction > 0 else 0
    n_train = n - n_test - n_val
    if n_train <= 0:
        raise ValueError(f"split fractions leave no training data (n={n}, "
                         f"val={n_val}, test={n_test})")
    test_idx = perm[:n_test]
    val_idx = perm[n_test:n_test + n_val]
    train_idx = perm[n_test + n_val:]
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def make_group_splits(
    groups: np.ndarray,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Deterministic partition that keeps whole groups together.

    Groups are shuffled with ``seed`` and assigned greedily: test first until
    it holds at least ``test_fraction`` of the rows, then val likewise, the
    rest is train. Each non-empty partition gets at least one group. Raises
    if there are not enough groups to populate every requested partition.
    """
    groups = np.asarray(groups).astype(str)
    n = len(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    n_groups = len(uniq)
    sizes = np.bincount(inv, minlength=n_groups)
    need = 1 + int(val_fraction > 0) + int(test_fraction > 0)
    if n_groups < need:
        raise ValueError(
            f"make_group_splits: {n_groups} group(s) cannot fill {need} partitions "
            f"(val_fraction={val_fraction}, test_fraction={test_fraction})"
        )
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_groups)

    part = np.full(n_groups, 0, dtype=np.int64)  # 0 train, 1 val, 2 test
    targets = {2: n * test_fraction, 1: n * val_fraction}
    filled = {2: 0, 1: 0}
    pos = 0
    for label in (2, 1):
        if targets[label] <= 0:
            continue
        # at least one group, then keep adding while below target and while
        # enough groups remain for the partitions still to fill
        remaining_needed = (1 if label == 2 and val_fraction > 0 else 0) + 1
        while pos < n_groups - remaining_needed and (
            filled[label] == 0 or filled[label] < targets[label]
        ):
            g = order[pos]
            part[g] = label
            filled[label] += int(sizes[g])
            pos += 1
    row_part = part[inv]
    splits = {
        "train": np.nonzero(row_part == 0)[0].astype(np.int64),
        "val": np.nonzero(row_part == 1)[0].astype(np.int64),
        "test": np.nonzero(row_part == 2)[0].astype(np.int64),
    }
    if len(splits["train"]) == 0:
        raise ValueError("make_group_splits: no training groups left")
    return splits


def save_splits(splits: dict[str, np.ndarray], path: str) -> None:
    """Persist a split dict to JSON for cross-script reproducibility."""
    out = {k: v.tolist() for k, v in splits.items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f)


def load_splits(path: str) -> dict[str, np.ndarray]:
    with open(path) as f:
        raw = json.load(f)
    return {k: np.asarray(v, dtype=np.int64) for k, v in raw.items()}


def get_or_make_splits(
    n: int,
    cache_dir: str,
    cache_key: str,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
    groups: np.ndarray | None = None,
    strategy: str = "random",
) -> tuple[dict[str, np.ndarray], str]:
    """Read split JSON if it matches (n, fractions, seed); otherwise create and
    save one. Returns (splits, path).

    ``strategy`` selects the partition rule:
        random           per-row permutation (file ``splits_<key>.json``, unchanged)
        group_sample     whole groups together, ``groups`` = sample ids
        group_instrument whole groups together, ``groups`` = instrument ids
    Grouped strategies require ``groups`` (length ``n``; see
    :func:`measured_groups`) and use ``splits_<key>_<strategy>.json``.
    """
    if strategy not in _SPLIT_STRATEGIES:
        raise ValueError(f"unknown split strategy {strategy!r}; choose from {_SPLIT_STRATEGIES}")
    if strategy == "random":
        splits_path = os.path.join(cache_dir, f"splits_{cache_key}.json")
    else:
        if groups is None:
            raise ValueError(f"split strategy {strategy!r} requires `groups`")
        groups = np.asarray(groups)
        if len(groups) != n:
            raise ValueError(f"`groups` has length {len(groups)}, expected n={n}")
        splits_path = os.path.join(cache_dir, f"splits_{cache_key}_{strategy}.json")
    if os.path.isfile(splits_path):
        splits = load_splits(splits_path)
        total = sum(len(v) for v in splits.values())
        if total == n:
            return splits, splits_path
        # Stale (n changed) — regenerate
    if strategy == "random":
        splits = make_splits(n, val_fraction, test_fraction, seed)
    else:
        splits = make_group_splits(groups, val_fraction, test_fraction, seed)
    save_splits(splits, splits_path)
    return splits, splits_path


def cluster_compositions(
    concentrations: np.ndarray,
    n_clusters: int,
    seed: int = 42,
) -> np.ndarray:
    """K-means class labels from concentration vectors. Used when the raw
    sample_type_id space is too granular (e.g. 2256 types) for classification.

    Returns int64 array of cluster IDs, shape [N]."""
    from sklearn.cluster import KMeans
    k = min(n_clusters, len(concentrations))
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    return km.fit_predict(concentrations).astype(np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: build a dataset from a YAML config (paths + ranges)
# ─────────────────────────────────────────────────────────────────────────────
def build_dataset_from_config(cfg: dict) -> Dataset:
    """Top-level entry point used by training scripts. `cfg` mirrors
    config/libs_data.yaml or config/libs_data_measured.yaml."""
    if cfg.get("source") == "measured":
        from data.measured_pipeline import build_measured_dataset_from_config
        return build_measured_dataset_from_config(cfg)

    paths = cfg["paths"]
    gen = cfg.get("generation", {})

    # physics_version 2 generator (Kirchhoff-consistent, physical optical depth)
    plasma_model = gen.get("plasma_model", "legacy")
    if plasma_model in _PLASMA_MODELS_V2:
        from data.two_zone_pipeline import build_two_zone_dataset_from_config  # lazy: avoids cycle
        return build_two_zone_dataset_from_config(cfg)
    if plasma_model != "legacy":
        raise ValueError(
            f"generation.plasma_model={plasma_model!r} not understood; "
            f"use 'legacy' or one of {_PLASMA_MODELS_V2}"
        )

    ranges = cfg["ranges"]

    db_path = str(Path(paths["db"]).expanduser().resolve())
    xlsx_path = str(Path(paths["sample_matrix"]).expanduser().resolve())
    wl_json = str(Path(paths["wavelength_json"]).expanduser().resolve())
    cache_dir = str(Path(paths.get("cache_dir", "external_data/cache")).expanduser().resolve())

    wavelength = load_wavelength(wl_json)
    sample_types = load_sample_types(xlsx_path, db_path)

    # Override n_samples per sample type if configured (otherwise stays at 1)
    n_per_type = gen.get("n_samples_per_type", 1)
    if n_per_type != 1:
        for st in sample_types:
            st["n_samples"] = n_per_type

    # Optionally cap the number of sample types (useful for smoke tests)
    max_types = gen.get("max_sample_types")
    if max_types is not None:
        sample_types = sample_types[:max_types]

    return SyntheticLIBSDataset(
        sample_types=sample_types,
        wavelength=wavelength,
        db_path=db_path,
        te_range=tuple(ranges["te"]),
        ne_range=tuple(ranges["ne"]),
        n_density=gen.get("number_density", DEFAULT_NUMBER_DENSITY),
        optical_path=gen.get("optical_path", DEFAULT_OPTICAL_PATH),
        n_workers=gen.get("n_workers", 1),
        cache_dir=cache_dir,
        seed=gen.get("seed", 42),
        verbose=gen.get("verbose", True),
    )
