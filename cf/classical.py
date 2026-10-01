"""
Zero-parameter ("classical") line weighting for the CF solver and the curated
54-line CF spark-OES list.

``classical_weights`` reproduces what a spectroscopist would do by hand:
use every line whose Voigt fit succeeded with R² ≥ ``r2_min`` and a positive
area, optionally restricted to isolated lines (``isolation_score`` ≥
``isolation_min`` unless the line was force-included).  The result is a
{0, 1} weight vector that ``saha_boltzmann_solve_np`` /
``SahaBoltzmannLayer`` accept directly, so the pure-physics variant of the
CF task differs from the learned one only in this vector.

``cf/data/cf_oes_lines_54.tsv`` holds the 54 fully-filled rows of
``Boltzmann_lines_v15.txt`` from the CF spark-OES project (Fe 19, Ni 9,
Cu 8, Mn 7, C 3, Cr 3, Si 3, Al 2); ``select_cf_oes_lines`` matches them to
a token cache by (element, stage, wavelength ± ``tol_nm``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.line_tokenization import atomic_number, ion_binary

CF_OES_54_TSV = Path(__file__).resolve().parent / "data" / "cf_oes_lines_54.tsv"
# REE-mineral LIBS-in-air list (air wavelengths, LIBS_data.db values); same column layout.
CF_MINERAL_TSV = Path(__file__).resolve().parent / "data" / "cf_mineral_lines.tsv"

# token channels (data/line_tokenization.py)
_CH_WL, _CH_Z, _CH_ION, _CH_AREA, _CH_R2 = 0, 7, 8, 9, 11


def classical_weights(
    tokens: np.ndarray,
    fit_valid: np.ndarray,
    r2_min: float = 0.9,
    min_area: float = 0.0,
    isolation: np.ndarray | None = None,
    forced: np.ndarray | None = None,
    isolation_min: float = 0.5,
) -> np.ndarray:
    """{0, 1} per-line weights from fit quality (and optional isolation).

    Args:
        tokens: ``[L, 14]`` (or ``[..., L, 14]``) raw line tokens
        fit_valid: ``[L]`` (or ``[..., L]``) Voigt-fit validity flags
        r2_min: minimum fitted R² (token channel 11)
        min_area: minimum fitted area (token channel 9, strict)
        isolation: optional ``[L]`` isolation score in [0, 1] (1 = isolated)
        forced: optional ``[L]`` flag; forced lines bypass the isolation cut
        isolation_min: isolation threshold applied when ``isolation`` is given

    Returns:
        float64 array with the leading shape of ``fit_valid``, values in {0, 1}.
    """
    tokens = np.asarray(tokens, dtype=np.float64)
    valid = np.asarray(fit_valid, dtype=np.float64) > 0.5
    area = tokens[..., _CH_AREA]
    r2 = tokens[..., _CH_R2]
    keep = valid & np.isfinite(area) & (area > float(min_area)) & (area > 0.0) & (r2 >= float(r2_min))
    if isolation is not None:
        iso_ok = np.asarray(isolation, dtype=np.float64) >= float(isolation_min)
        if forced is not None:
            iso_ok = iso_ok | (np.asarray(forced, dtype=np.float64) > 0.5)
        keep = keep & np.broadcast_to(iso_ok, keep.shape)
    return keep.astype(np.float64)


def load_cf_oes_lines(tsv_path: str | Path | None = None) -> pd.DataFrame:
    """Read the curated list (comment lines start with ``#``).  Adds integer
    columns ``Z`` and ``ion`` (0 = I, 1 = II) and float ``Wl``."""
    path = Path(tsv_path) if tsv_path is not None else CF_OES_54_TSV
    df = pd.read_csv(path, sep="\t", comment="#", dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    df["Elem_name"] = df["Elem_name"].str.strip()
    df["Z"] = df["Elem_name"].map(atomic_number).astype(np.int64)
    df["ion"] = df["ion.state"].map(ion_binary).astype(np.int64)
    for c in ("Wl", "Ei", "Ek", "Ak", "gi", "gk"):
        df[c] = df[c].astype(np.float64)
    return df


def select_cf_oes_lines(
    tokens_central_wavelength: np.ndarray,
    tokens_Z: np.ndarray,
    tokens_ion: np.ndarray,
    tsv_path: str | Path | None = None,
    tol_nm: float = 0.01,
) -> np.ndarray:
    """Boolean ``[L]`` mask of token lines present in the curated list.

    A token matches when its element (Z) and stage (ion_binary) agree and
    ``|wavelength - Wl| <= tol_nm``.  If several tokens match one curated
    line, the closest one is taken.
    """
    wl = np.asarray(tokens_central_wavelength, dtype=np.float64).reshape(-1)
    Z = np.asarray(np.rint(tokens_Z), dtype=np.int64).reshape(-1)
    ion = np.asarray(np.rint(tokens_ion), dtype=np.int64).reshape(-1)
    ref = load_cf_oes_lines(tsv_path)
    mask = np.zeros(wl.shape[0], dtype=bool)
    for _, row in ref.iterrows():
        cand = np.flatnonzero((Z == row["Z"]) & (ion == row["ion"]) & (np.abs(wl - row["Wl"]) <= tol_nm))
        if cand.size:
            mask[cand[np.argmin(np.abs(wl[cand] - row["Wl"]))]] = True
    return mask
