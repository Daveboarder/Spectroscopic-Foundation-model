"""
Theoretical spectral-line dictionary over a Te × Ne grid.

Computes per-line optically thin integrated intensities (no Voigt broadening)
with the Kirchhoff-consistent physics of ``data.plasma_physics``, keeps the
maximum over the grid per line, selects lines per element and caches to HDF5.

Selection modes (``line_dictionary.selection.mode``):

* ``top_percent_per_element`` (legacy default) — strongest ``percent`` % per element.
* ``threshold`` (legacy) — absolute threshold on the grid-max intensity.
* ``cf_isolated`` — calibration-free line list: candidates are restricted to
  the target elements (default: the element columns of the sample matrix),
  scored for spectral isolation against every line of every matrix element
  (weighted by its maximum concentration in the matrix) plus ambient gas lines
  (e.g. Ar), and a curated ``force_include`` list is always kept.  The HDF5
  gains ``isolation_score`` [n_lines] float32 (1 = isolated) and ``forced``
  [n_lines] uint8 datasets (contract C4).

The intensity stored as ``theoretical_intensity`` (token channel 6) is the
thin-limit integrated radiance per unit ``x_e * N * l`` (number fraction times
heavy-particle density times path length), i.e. ``thin_line_intensities`` with
``x_e = 1, N = 1, l = 1``.  Legacy config keys ``N``, ``C``, ``l`` are ignored
(they only rescaled the intensities in the old, incorrect emissivity).
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from data.atomic_data import mass_to_number_fractions
from data.libs_pipeline import (
    is_element_symbol,
    line_db_cache_key,
    load_sample_types,
    load_wavelength,
)
from data.plasma_physics import thin_line_intensities

# Ion state vocabulary for categorical embedding
ION_STATES = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
ION_STATE_TO_ID = {s: i for i, s in enumerate(ION_STATES)}

# Bumped when the physics behind ``theoretical_intensity`` changes so that
# cache keys (which otherwise only hash the config) do not collide with
# dictionaries built by an older emissivity formula.
LINE_DICT_PHYSICS_VERSION = 2

# Wavelength tolerance for matching curated (force_include) lines to DB rows.
FORCE_INCLUDE_TOL_NM = 0.01

_SELECTION_MODES = ("top_percent_per_element", "threshold", "cf_isolated")


def _config_hash(cfg: dict) -> str:
    return hashlib.md5(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _te_ne_grid(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    te_lo, te_hi = float(cfg["te_range"][0]), float(cfg["te_range"][1])
    ne_lo, ne_hi = float(cfg["ne_range"][0]), float(cfg["ne_range"][1])
    n_te = int(cfg["n_grid"]["te"])
    n_ne = int(cfg["n_grid"]["ne"])
    te_grid = np.linspace(te_lo, te_hi, n_te)
    if cfg.get("ne_log_spaced", True):
        ne_grid = np.logspace(np.log10(ne_lo), np.log10(ne_hi), n_ne)
    else:
        ne_grid = np.linspace(ne_lo, ne_hi, n_ne)
    return te_grid, ne_grid


def compute_line_intensities_for_plasma(
    element: str,
    Te: float,
    Ne: float,
    N: float,
    C: float,
    l: float,
    db_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-line optically thin integrated intensities for one (Te, Ne) point.

    Thin wrapper over :func:`data.plasma_physics.thin_line_intensities` kept
    for the legacy call signature: the intensity is linear in
    ``C * N * l`` (number fraction × heavy-particle density [cm^-3] × path [cm]).
    The dictionary builder calls it with ``N = C = l = 1``.

    Returns:
        wl [nm], ion_state_str, Ei [eV], Ek [eV], gi, gk, Ak [s^-1],
        intensity [erg s^-1 cm^-2 sr^-1 per unit C*N*l]
    """
    res = thin_line_intensities(element, float(Te), float(Ne), float(C), float(N), float(l), db_path)
    if res["wl"].size == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty, empty, empty, empty, empty, empty
    return (
        res["wl"], res["ion_state"], res["Ei"], res["Ek"], res["gi"], res["gk"], res["Ak"],
        np.asarray(res["I_int"], dtype=np.float64),
    )


def _list_elements(db_path: str) -> list[str]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT Elem_name FROM QuantParam ORDER BY Elem_name")
    # Skip DB artefacts ("Al-II", "", "n", "r" in LIBS_data.db): they have no E_ion entry.
    elems = [str(r[0]).strip() for r in cur.fetchall() if is_element_symbol(r[0])]
    conn.close()
    return elems


def _wavelength_clip_bounds(cfg: dict, project_root: Path) -> tuple[float | None, float | None]:
    clip = cfg.get("wavelength_clip") or {}
    wmin, wmax = clip.get("min"), clip.get("max")
    if wmin is not None and wmax is not None:
        return float(wmin), float(wmax)
    if cfg.get("use_wavelength_json_clip") and cfg.get("wavelength_json"):
        wl_path = project_root / cfg["wavelength_json"]
        if wl_path.is_file():
            wl = load_wavelength(str(wl_path))
            return float(wl.min()), float(wl.max())
    return wmin, wmax


def _select_lines_for_element(
    element_records: list[dict[str, Any]],
    selection_cfg: dict[str, Any],
    threshold: float,
) -> list[dict[str, Any]]:
    """Select lines for one element based on config mode (legacy modes only;
    ``cf_isolated`` is handled by :func:`_apply_isolation_filter`)."""
    mode = str(selection_cfg.get("mode", "top_percent_per_element")).strip().lower()
    if mode == "threshold":
        return [r for r in element_records if r["theoretical_intensity"] >= threshold]
    if mode == "cf_isolated":
        # Already selected by the isolation post-pass.
        return list(element_records)
    if mode != "top_percent_per_element":
        raise ValueError(
            f"Unknown line_dictionary.selection.mode='{mode}'. "
            f"Expected one of {_SELECTION_MODES}."
        )

    percent = float(selection_cfg.get("percent", 10.0))
    min_keep = int(selection_cfg.get("min_keep", 10))
    if percent <= 0.0:
        raise ValueError("line_dictionary.selection.percent must be > 0")
    if min_keep < 1:
        raise ValueError("line_dictionary.selection.min_keep must be >= 1")

    n = len(element_records)
    if n < min_keep:
        k = n
    else:
        k = max(1, int(math.ceil((percent / 100.0) * n)))
    return sorted(element_records, key=lambda r: r["theoretical_intensity"], reverse=True)[:k]


# ─────────────────────────────────────────────────────────────────────────────
# Candidate collection (grid maximum per line)
# ─────────────────────────────────────────────────────────────────────────────
def _collect_candidates(
    elem: str,
    te_grid: np.ndarray,
    ne_grid: np.ndarray,
    db_path: str,
    wmin: float | None,
    wmax: float | None,
) -> list[dict[str, Any]]:
    """All DB lines of ``elem`` inside the wavelength clip with their
    grid-maximum thin-limit intensity (per unit x_e·N·l) and the (Te, Ne)
    at which it is attained.  Lines are unique per (wavelength, ion_state)."""
    wl = ion = Ei = Ek = gi = gk = Ak = None
    best_I: np.ndarray | None = None
    best_te: np.ndarray | None = None
    best_ne: np.ndarray | None = None
    for Te in te_grid:
        for Ne in ne_grid:
            wl, ion, Ei, Ek, gi, gk, Ak, I = compute_line_intensities_for_plasma(
                elem, float(Te), float(Ne), 1.0, 1.0, 1.0, db_path,
            )
            if wl.size == 0:
                return []
            if best_I is None:
                best_I = I.copy()
                best_te = np.full(wl.size, float(Te))
                best_ne = np.full(wl.size, float(Ne))
            else:
                better = I > best_I
                best_I = np.where(better, I, best_I)
                best_te = np.where(better, float(Te), best_te)
                best_ne = np.where(better, float(Ne), best_ne)
    if wl is None or best_I is None:
        return []

    keep = np.ones(wl.size, dtype=bool)
    if wmin is not None:
        keep &= wl >= wmin
    if wmax is not None:
        keep &= wl <= wmax

    best: dict[tuple[float, str], dict[str, Any]] = {}
    for i in np.flatnonzero(keep):
        key_line = (float(wl[i]), str(ion[i]))
        prev = best.get(key_line)
        if prev is None or best_I[i] > prev["theoretical_intensity"]:
            best[key_line] = {
                "central_wavelength": float(wl[i]),
                "element": elem,
                "ion_state": str(ion[i]),
                "Ei": float(Ei[i]),
                "Ek": float(Ek[i]),
                "gi": float(gi[i]),
                "gk": float(gk[i]),
                "Ak": float(Ak[i]),
                "theoretical_intensity": float(best_I[i]),
                "Te_opt": float(best_te[i]),
                "Ne_opt": float(best_ne[i]),
            }
    return list(best.values())


# ─────────────────────────────────────────────────────────────────────────────
# cf_isolated helpers
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_path(path: str | Path, project_root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (project_root / p)


def resolve_target_elements(
    ld_cfg: dict[str, Any], project_root: Path, db_path: str,
) -> list[str]:
    """Elements the CF dictionary is built for.

    ``selection.target_elements`` is either ``"from_sample_matrix"`` (element
    columns of ``selection.sample_matrix`` that exist in the DB, i.e. the 38
    steel-matrix elements), ``"all"`` (every DB element) or an explicit list.
    """
    sel = dict(ld_cfg.get("selection", {}))
    spec = sel.get("target_elements", "from_sample_matrix")
    db_elems = set(_list_elements(db_path))
    if isinstance(spec, str):
        spec_l = spec.strip().lower()
        if spec_l == "all":
            return sorted(db_elems)
        if spec_l != "from_sample_matrix":
            raise ValueError(
                "line_dictionary.selection.target_elements must be 'from_sample_matrix', "
                f"'all' or a list of symbols; got {spec!r}"
            )
        matrix = sel.get("sample_matrix")
        if not matrix:
            raise ValueError(
                "selection.target_elements='from_sample_matrix' requires selection.sample_matrix"
            )
        samples = load_sample_types(str(_resolve_path(matrix, project_root)), db_path)
        if not samples:
            raise RuntimeError(f"Sample matrix {matrix} has no rows.")
        return [e for e in samples[0]["concentration_ranges"].keys() if e in db_elems]
    targets = [str(e).strip() for e in spec]
    missing = [e for e in targets if e not in db_elems]
    if missing:
        raise ValueError(f"target_elements not in the line DB: {missing}")
    return targets


def _interferer_weights(
    sel: dict[str, Any], project_root: Path, db_path: str,
) -> dict[str, float]:
    """Per-element interferer weight: the maximum concentration of the element
    over all sample-matrix rows (upper uncertainty bound, row-normalised),
    expressed as a **number** fraction (``selection.interferer_units: number``,
    default) or a mass fraction (``mass``); ``selection.ambient_elements``
    (e.g. ``{Ar: 1.0}``) are merged in with their given weight."""
    weights: dict[str, float] = {}
    matrix = sel.get("sample_matrix")
    units = str(sel.get("interferer_units", "number")).strip().lower()
    if units not in ("number", "mass"):
        raise ValueError(f"selection.interferer_units must be 'number' or 'mass', got {units!r}")
    if matrix:
        samples = load_sample_types(str(_resolve_path(matrix, project_root)), db_path)
        if samples:
            elems = list(samples[0]["concentration_ranges"].keys())
            W = np.array(
                [[max(0.0, float(s["concentration_ranges"][e][1])) for e in elems] for s in samples],
                dtype=np.float64,
            )
            row_sum = W.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0] = 1.0
            W = W / row_sum
            X = mass_to_number_fractions(W, elems) if units == "number" else W
            weights = {e: float(X[:, j].max()) for j, e in enumerate(elems)}
    for e, w in (sel.get("ambient_elements") or {}).items():
        weights[str(e)] = max(weights.get(str(e), 0.0), float(w))
    return weights


def _normalise_stage(value: Any) -> str:
    s = str(value).strip().upper()
    if s in ("1", "1.0", "0"):
        return "I"
    if s in ("2", "2.0"):
        return "II"
    return s


def load_force_include(path: str | Path) -> list[dict[str, Any]]:
    """Read a curated line list (TSV/CSV) into ``[{element, ion_state, wavelength}]``.

    Accepts the ``cf/data/cf_oes_lines_54.tsv`` layout as well as the raw
    ``Boltzmann_lines_v15.txt`` of the CF_OES project (columns ``Elem_name,
    ion.state, Wl, ...``); for the raw file only rows with all columns filled
    (lineLeft/lineRight/bgrLeft/bgrRight/coment) are used.
    """
    path = Path(path)
    sep = "," if path.suffix.lower() == ".csv" else "\t"
    df = pd.read_csv(path, sep=sep, comment="#")
    cols = {str(c).strip().lower(): c for c in df.columns}

    def pick(*names: str) -> str:
        for n in names:
            if n in cols:
                return cols[n]
        raise ValueError(f"{path}: none of the columns {names} found (have {list(df.columns)})")

    c_elem = pick("element", "elem_name", "elem", "el", "symbol")
    c_stage = pick("ion_state", "ion.state", "stage", "ion", "ionstate", "ion_stage")
    c_wl = pick("wavelength", "wl", "central_wavelength", "wavelength_nm", "lambda", "lambda_nm")

    raw_cols = [cols[c] for c in ("lineleft", "lineright", "bgrleft", "bgrright", "coment") if c in cols]
    if raw_cols:
        df = df.dropna(subset=raw_cols)

    out: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        wl = pd.to_numeric(row[c_wl], errors="coerce")
        if pd.isna(wl):
            continue
        out.append({
            "element": str(row[c_elem]).strip(),
            "ion_state": _normalise_stage(row[c_stage]),
            "wavelength": float(wl),
        })
    return out


def _reference_line_table(
    elements: list[str], Te: float, Ne: float, db_path: str,
) -> dict[str, dict[str, np.ndarray]]:
    """Thin-limit intensities per unit x_e·N·l at the reference plasma for
    every element in ``elements`` (deduplicated per (wavelength, stage))."""
    table: dict[str, dict[str, np.ndarray]] = {}
    for elem in elements:
        wl, ion, _, _, _, _, _, I = compute_line_intensities_for_plasma(
            elem, Te, Ne, 1.0, 1.0, 1.0, db_path,
        )
        if wl.size == 0:
            continue
        df = pd.DataFrame({"wl": wl, "ion": np.asarray(ion).astype(str), "I": I})
        df = df.groupby(["wl", "ion"], as_index=False)["I"].max()
        table[elem] = {
            "wl": df["wl"].values.astype(np.float64),
            "ion": df["ion"].values.astype(str),
            "I": df["I"].values.astype(np.float64),
        }
    return table


def _apply_isolation_filter(
    records_all: dict[str, list[dict[str, Any]]],
    cfg: dict[str, Any],
    project_root: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """``cf_isolated`` selection post-pass (runs before per-element selection).

    Args:
        records_all: candidate records per target element (output of
            :func:`_collect_candidates`, i.e. inside the wavelength clip).
        cfg: the ``line_dictionary`` config section (a full line-embedding
            config with a ``line_dictionary`` key is also accepted).
        project_root: base for relative paths (``db_path``, ``sample_matrix``,
            ``force_include``).

    Every candidate gets ``isolation_score``, ``forced`` and
    ``reference_intensity`` fields.  For candidate *i* of element *e* at the
    reference plasma (``reference_te``, ``reference_ne``):

        interference_i = sum_{j != i, |λ_j − λ_i| <= isolation_window_nm} I_j · C_j^max
        ratio_i        = interference_i / (I_i · C_e^typ)
        isolated_i     = ratio_i <= max_interference_fraction
        isolation_score_i = clip(1 − ratio_i, 0, 1)

    where the sum runs over all lines of all matrix elements (weighted by
    their maximum matrix concentration) plus ``ambient_elements`` (e.g. Ar,
    taken from the DB although absent from the matrix), including other lines
    of the candidate's own element; ``C_e^typ`` is the candidate element's own
    maximum concentration (floored at ``typical_concentration_floor``).

    Per element the isolated lines are ranked by ``theoretical_intensity`` and
    the top ``max_lines_per_species`` are kept; if fewer than
    ``min_lines_per_species`` are isolated, the best non-isolated lines (by
    score, then intensity) fill up — keeping their low score.  Rows of the
    ``force_include`` list matched by (element, stage, |Δλ| <= 0.01 nm) are
    always kept and flagged ``forced = 1`` even when not isolated.

    Returns:
        (selected records per element, report dict with per-element counts,
        force-include matches and the score distribution).
    """
    if "line_dictionary" in cfg and "selection" not in cfg:
        cfg = cfg["line_dictionary"]
    sel = dict(cfg.get("selection", {}))
    db_path = str(_resolve_path(cfg["db_path"], project_root))

    window = float(sel.get("isolation_window_nm", 0.3))
    max_frac = float(sel.get("max_interference_fraction", 0.1))
    ref_te = float(sel.get("reference_te", 10000.0))
    ref_ne = float(sel.get("reference_ne", 1.0e17))
    max_per = int(sel.get("max_lines_per_species", 40))
    min_per = int(sel.get("min_lines_per_species", 1))
    c_floor = float(sel.get("typical_concentration_floor", 1.0e-4))
    if window <= 0 or max_frac < 0 or max_per < 1 or min_per < 0:
        raise ValueError("Invalid cf_isolated selection parameters "
                         f"(window={window}, max_frac={max_frac}, max={max_per}, min={min_per})")
    min_per = min(min_per, max_per)

    targets = list(records_all.keys())
    weights = _interferer_weights(sel, project_root, db_path)
    # Every target contributes self-blends even if it is absent from the matrix.
    ref_elements = sorted(set(targets) | {e for e, w in weights.items() if w > 0})
    ref = _reference_line_table(ref_elements, ref_te, ref_ne, db_path)
    missing_ref = [e for e in ref_elements if e not in ref]
    if missing_ref:
        raise RuntimeError(f"No DB lines for interferer/target elements {missing_ref}")

    # Global weighted interferer table sorted by wavelength.
    wl_all = np.concatenate([ref[e]["wl"] for e in ref_elements])
    wI_all = np.concatenate([ref[e]["I"] * weights.get(e, 0.0) for e in ref_elements])
    order = np.argsort(wl_all, kind="stable")
    wl_sorted, wI_sorted = wl_all[order], wI_all[order]
    cum = np.concatenate([[0.0], np.cumsum(wI_sorted)])

    # Curated lines.
    force_path = sel.get("force_include")
    forced_rows: list[dict[str, Any]] = []
    if force_path:
        fp = _resolve_path(force_path, project_root)
        if not fp.is_file():
            raise FileNotFoundError(f"selection.force_include file not found: {fp}")
        forced_rows = load_force_include(fp)
    forced_by_elem: dict[str, list[dict[str, Any]]] = {}
    for row in forced_rows:
        forced_by_elem.setdefault(row["element"], []).append(row)

    selected: dict[str, list[dict[str, Any]]] = {}
    per_elem: dict[str, dict[str, Any]] = {}
    forced_matched: list[dict[str, Any]] = []
    forced_unmatched: list[dict[str, Any]] = []
    forced_skipped = [r for e, rows in forced_by_elem.items() if e not in records_all for r in rows]

    for elem in targets:
        recs = records_all[elem]
        c_typ = max(weights.get(elem, 0.0), c_floor)
        own = {(float(w), str(s)): float(v) for w, s, v in zip(ref[elem]["wl"], ref[elem]["ion"], ref[elem]["I"])}
        for r in recs:
            wl_i = r["central_wavelength"]
            I_ref = own.get((wl_i, r["ion_state"]), 0.0)
            lo = int(np.searchsorted(wl_sorted, wl_i - window, side="left"))
            hi = int(np.searchsorted(wl_sorted, wl_i + window, side="right"))
            interference = float(cum[hi] - cum[lo]) - I_ref * weights.get(elem, 0.0)
            interference = max(interference, 0.0)
            denom = I_ref * c_typ
            ratio = interference / denom if denom > 0 else np.inf
            r["reference_intensity"] = I_ref
            r["interference_ratio"] = float(ratio)
            r["isolation_score"] = float(np.clip(1.0 - ratio, 0.0, 1.0)) if np.isfinite(ratio) else 0.0
            r["isolated"] = bool(ratio <= max_frac)
            r["forced"] = 0

        # Force-include matching by (element, stage, |Δλ| <= tol).
        for row in forced_by_elem.get(elem, []):
            cands = [
                r for r in recs
                if r["ion_state"] == row["ion_state"]
                and abs(r["central_wavelength"] - row["wavelength"]) <= FORCE_INCLUDE_TOL_NM
            ]
            if not cands:
                forced_unmatched.append(row)
                continue
            best = min(cands, key=lambda r: abs(r["central_wavelength"] - row["wavelength"]))
            best["forced"] = 1
            forced_matched.append({**row, "matched_wavelength": best["central_wavelength"]})

        forced = [r for r in recs if r["forced"]]
        isolated = sorted(
            (r for r in recs if r["isolated"] and not r["forced"]),
            key=lambda r: r["theoretical_intensity"], reverse=True,
        )
        rest = sorted(
            (r for r in recs if not r["isolated"] and not r["forced"]),
            key=lambda r: (r["isolation_score"], r["theoretical_intensity"]), reverse=True,
        )
        chosen = list(forced)
        room = max(0, max_per - len(chosen))
        chosen.extend(isolated[:room])
        if len(chosen) < min_per:
            chosen.extend(rest[: min_per - len(chosen)])
        selected[elem] = chosen
        ratios = np.array([r["interference_ratio"] for r in recs], dtype=np.float64)
        per_elem[elem] = {
            "n_candidates": len(recs),
            "n_isolated": int(sum(1 for r in recs if r["isolated"])),
            "n_selected": len(chosen),
            "n_forced": len(forced),
            "n_forced_isolated": int(sum(1 for r in forced if r["isolated"])),
            "n_filled_non_isolated": max(0, len(chosen) - len(forced) - min(len(isolated), room)),
            "n_ratio_below_0.5": int(np.sum(ratios <= 0.5)) if ratios.size else 0,
            "n_ratio_below_1": int(np.sum(ratios <= 1.0)) if ratios.size else 0,
            "min_ratio": float(ratios.min()) if ratios.size else None,
            "c_typ": c_typ,
        }

    scores = np.array([r["isolation_score"] for recs in selected.values() for r in recs], dtype=np.float64)
    report = {
        "reference_te": ref_te,
        "reference_ne": ref_ne,
        "isolation_window_nm": window,
        "max_interference_fraction": max_frac,
        "interferer_weights": weights,
        "per_element": per_elem,
        "n_forced_rows": len(forced_rows),
        "n_forced_matched": len(forced_matched),
        "forced_matched": forced_matched,
        "forced_unmatched": forced_unmatched,
        "forced_skipped_not_target": forced_skipped,
        "n_selected": int(scores.size),
        "n_isolated_selected": int(np.sum(scores >= 1.0 - max_frac)) if scores.size else 0,
        "score_quantiles": (
            {q: float(np.quantile(scores, q)) for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)}
            if scores.size else {}
        ),
    }
    return selected, report


def _print_isolation_report(report: dict[str, Any]) -> None:
    pe = report["per_element"]
    print(
        f"  cf_isolated: reference plasma Te={report['reference_te']:.0f} K, "
        f"Ne={report['reference_ne']:.2e} cm^-3, window ±{report['isolation_window_nm']} nm, "
        f"max interference {report['max_interference_fraction']:.2f}"
    )
    summary = ", ".join(
        f"{e}:{pe[e]['n_selected']}/{pe[e]['n_isolated']}/{pe[e]['n_candidates']}"
        + (f"(f{pe[e]['n_forced']})" if pe[e]["n_forced"] else "")
        for e in sorted(pe)
    )
    print(f"  Kept per element (selected/isolated/candidates, f=forced): {summary}")
    n_iso_50 = sum(v["n_ratio_below_0.5"] for v in pe.values())
    n_iso_100 = sum(v["n_ratio_below_1"] for v in pe.values())
    print(
        f"  Candidates isolated at max_interference_fraction={report['max_interference_fraction']:.2f}: "
        f"{sum(v['n_isolated'] for v in pe.values())}; at 0.5: {n_iso_50}; at 1.0: {n_iso_100}"
    )
    print(
        f"  force_include: {report['n_forced_matched']}/{report['n_forced_rows']} rows matched"
        + (f", {len(report['forced_unmatched'])} unmatched" if report["forced_unmatched"] else "")
        + (f", {len(report['forced_skipped_not_target'])} not a target element"
           if report["forced_skipped_not_target"] else "")
    )
    for row in report["forced_unmatched"][:20]:
        print(f"    unmatched: {row['element']} {row['ion_state']} {row['wavelength']:.3f} nm")
    q = report["score_quantiles"]
    if q:
        print(
            "  isolation_score quantiles: "
            + ", ".join(f"q{int(k * 100):02d}={v:.3f}" for k, v in q.items())
            + f"; isolated {report['n_isolated_selected']}/{report['n_selected']}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Builder
# ─────────────────────────────────────────────────────────────────────────────
def _hash_config(ld_cfg: dict[str, Any], project_root: Path) -> dict[str, Any]:
    """Config subset that determines the cache key (adds physics version and
    the content hash of the force_include file when present)."""
    hash_cfg = {k: v for k, v in ld_cfg.items() if k != "cache_dir"}
    hash_cfg["physics_version"] = LINE_DICT_PHYSICS_VERSION
    # Content hash of the line DB (not for the legacy vacuum DB: keeps its old hashes).
    hash_cfg.update(line_db_cache_key(str(_resolve_path(ld_cfg["db_path"], project_root))))
    sel = ld_cfg.get("selection") or {}
    force_path = sel.get("force_include")
    if str(sel.get("mode", "")).strip().lower() == "cf_isolated" and force_path:
        fp = _resolve_path(force_path, project_root)
        if not fp.is_file():
            raise FileNotFoundError(
                f"line_dictionary.selection.force_include file not found: {fp}"
            )
        hash_cfg["force_include_md5"] = hashlib.md5(fp.read_bytes()).hexdigest()[:12]
    return hash_cfg


def build_line_dictionary(cfg: dict, project_root: Path | None = None, verbose: bool = True) -> str:
    """
    Build or load cached line dictionary HDF5.

    Returns:
        Path to the cache file.
    """
    project_root = project_root or Path(__file__).resolve().parents[1]
    ld_cfg = dict(cfg["line_dictionary"])
    db_path = str((project_root / ld_cfg["db_path"]).resolve())
    cache_dir = project_root / ld_cfg.get("cache_dir", "external_data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    hash_cfg = _hash_config(ld_cfg, project_root)
    key = _config_hash(hash_cfg)
    out_path = cache_dir / f"line_dict_{key}.h5"

    if out_path.is_file():
        if verbose:
            print(f"Line dictionary cache hit: {out_path}")
        return str(out_path)

    te_grid, ne_grid = _te_ne_grid(ld_cfg)
    threshold = float(ld_cfg.get("intensity_threshold", 0.0))
    selection_cfg = dict(ld_cfg.get("selection", {}))
    if "mode" not in selection_cfg:
        selection_cfg["mode"] = "top_percent_per_element"
    if "percent" not in selection_cfg:
        selection_cfg["percent"] = 10.0
    if "min_keep" not in selection_cfg:
        selection_cfg["min_keep"] = 10
    mode = str(selection_cfg["mode"]).strip().lower()
    if mode not in _SELECTION_MODES:
        raise ValueError(
            f"Unknown line_dictionary.selection.mode='{mode}'. Expected one of {_SELECTION_MODES}."
        )
    wmin, wmax = _wavelength_clip_bounds(ld_cfg, project_root)

    if mode == "cf_isolated":
        elements = resolve_target_elements(ld_cfg, project_root, db_path)
    else:
        elements = _list_elements(db_path)
    if verbose:
        print(f"Building line dictionary: {len(elements)} elements, "
              f"Te×Ne = {len(te_grid)}×{len(ne_grid)}, mode={mode}")

    records_all: dict[str, list[dict[str, Any]]] = {}
    for elem in elements:
        records_all[elem] = _collect_candidates(elem, te_grid, ne_grid, db_path, wmin, wmax)
    total_by_element = {e: len(r) for e, r in records_all.items()}

    records: list[dict[str, Any]] = []
    kept_by_element: dict[str, int] = {}
    report: dict[str, Any] | None = None
    if mode == "cf_isolated":
        selected_by_elem, report = _apply_isolation_filter(records_all, ld_cfg, project_root)
        for elem, sel_recs in selected_by_elem.items():
            kept_by_element[elem] = len(sel_recs)
            records.extend(sel_recs)
    else:
        for elem, elem_records in records_all.items():
            selected = _select_lines_for_element(
                element_records=elem_records,
                selection_cfg=selection_cfg,
                threshold=threshold,
            )
            for r in selected:
                r["isolation_score"] = 1.0
                r["forced"] = 0
                r["reference_intensity"] = r["theoretical_intensity"]
            kept_by_element[elem] = len(selected)
            records.extend(selected)

    if not records:
        raise RuntimeError(
            f"No lines selected for mode '{mode}'. "
            "Adjust line_dictionary.selection or intensity_threshold."
        )

    df = pd.DataFrame(records).sort_values("central_wavelength").reset_index(drop=True)
    max_lines = ld_cfg.get("max_lines")
    if max_lines is not None and len(df) > int(max_lines):
        forced_df = df[df["forced"] > 0]
        free_df = df[df["forced"] == 0]
        n_free = max(0, int(max_lines) - len(forced_df))
        df = (
            pd.concat([forced_df, free_df.nlargest(n_free, "theoretical_intensity")])
            .sort_values("central_wavelength")
            .reset_index(drop=True)
        )
        if verbose:
            print(f"  Subsampled to max_lines={max_lines} (forced lines kept)")
    elem_to_id = {e: i for i, e in enumerate(sorted(df["element"].unique()))}
    df["element_id"] = df["element"].map(elem_to_id)
    df["ion_state_id"] = df["ion_state"].map(lambda s: ION_STATE_TO_ID.get(s, 0))

    if verbose:
        if mode == "threshold":
            print(f"  Kept {len(df)} lines (threshold={threshold})")
        elif mode == "cf_isolated":
            print(f"  Kept {len(df)} lines (mode={mode}, {len(elem_to_id)} elements)")
            if report is not None:
                _print_isolation_report(report)
        else:
            pct = float(selection_cfg.get("percent", 10.0))
            min_keep = int(selection_cfg.get("min_keep", 10))
            print(
                f"  Kept {len(df)} lines (mode={mode}, percent={pct}, min_keep={min_keep})"
            )
            summary = ", ".join(
                f"{e}:{kept_by_element.get(e, 0)}/{total_by_element.get(e, 0)}"
                for e in sorted(total_by_element)
            )
            print(f"  Kept per element (kept/total): {summary}")

    str_dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(out_path, "w") as f:
        f.attrs["config_hash"] = key
        f.attrs["n_lines"] = len(df)
        f.attrs["config_json"] = json.dumps(hash_cfg, sort_keys=True)
        f.attrs["physics_version"] = LINE_DICT_PHYSICS_VERSION
        f.attrs["selection_mode"] = mode
        f.attrs["target_elements"] = json.dumps(list(elements))
        if report is not None:
            f.attrs["isolation_report"] = json.dumps(
                {k: v for k, v in report.items() if k not in ("forced_matched",)},
                sort_keys=True, default=str,
            )
        f.create_dataset("central_wavelength", data=df["central_wavelength"].values)
        f.create_dataset("theoretical_intensity", data=df["theoretical_intensity"].values)
        f.create_dataset("reference_intensity", data=df["reference_intensity"].values.astype(np.float64))
        f.create_dataset("isolation_score", data=df["isolation_score"].values.astype(np.float32))
        f.create_dataset("forced", data=df["forced"].values.astype(np.uint8))
        f.create_dataset("Te_opt", data=df["Te_opt"].values)
        f.create_dataset("Ne_opt", data=df["Ne_opt"].values)
        f.create_dataset("Ei", data=df["Ei"].values)
        f.create_dataset("Ek", data=df["Ek"].values)
        f.create_dataset("gi", data=df["gi"].values)
        f.create_dataset("gk", data=df["gk"].values)
        f.create_dataset("Ak", data=df["Ak"].values)
        f.create_dataset("element_id", data=df["element_id"].values.astype(np.int32))
        f.create_dataset("ion_state_id", data=df["ion_state_id"].values.astype(np.int32))
        g = f.create_group("vocab")
        g.attrs["elements"] = json.dumps(elem_to_id)
        g.attrs["ion_states"] = json.dumps(ION_STATE_TO_ID)
        g.create_dataset("element", data=df["element"].astype(str).values, dtype=str_dt)
        g.create_dataset("ion_state", data=df["ion_state"].astype(str).values, dtype=str_dt)

    if verbose:
        print(f"Saved line dictionary: {out_path}")
    return str(out_path)


def load_line_dictionary_meta(path: str) -> dict[str, Any]:
    """Load dictionary arrays and vocab without keeping the file open.

    ``isolation_score`` / ``forced`` are filled with ones / zeros for
    dictionaries written before the ``cf_isolated`` mode existed.
    """
    with h5py.File(path, "r") as f:
        vocab = json.loads(f["vocab"].attrs["elements"])
        n_lines = int(f.attrs["n_lines"])
        meta = {
            "path": path,
            "n_lines": n_lines,
            "config_hash": f.attrs.get("config_hash", ""),
            "physics_version": int(f.attrs.get("physics_version", 1)),
            "selection_mode": str(f.attrs.get("selection_mode", "")),
            "central_wavelength": f["central_wavelength"][:],
            "theoretical_intensity": f["theoretical_intensity"][:],
            "Ei": f["Ei"][:],
            "Ek": f["Ek"][:],
            "gi": f["gi"][:],
            "gk": f["gk"][:],
            "Ak": f["Ak"][:],
            "element_id": f["element_id"][:],
            "ion_state_id": f["ion_state_id"][:],
            "isolation_score": (
                f["isolation_score"][:].astype(np.float32) if "isolation_score" in f
                else np.ones(n_lines, dtype=np.float32)
            ),
            "forced": (
                f["forced"][:].astype(np.uint8) if "forced" in f
                else np.zeros(n_lines, dtype=np.uint8)
            ),
            "element_vocab": vocab,
            "n_elements": len(vocab),
        }
    return meta


if __name__ == "__main__":
    import argparse

    import yaml

    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build (or reuse) the line dictionary cache.")
    parser.add_argument("--config", type=str, default="config/line_embedding.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(root / args.config))
    build_line_dictionary(cfg, project_root=root, verbose=True)
