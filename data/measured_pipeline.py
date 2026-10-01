"""
Measured LIBS spectrum pipeline.

Reads experimental Chameleon/OptiCal JSON spectra, joins concentrations from
the sample matrix by filename stem, and caches to HDF5 in the same schema as
:data.libs_pipeline.SyntheticLIBSDataset.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from data.libs_pipeline import (
    _db_elements,
    _normalize_sample_id,
    compute_ccd_wavelengths,
    line_db_cache_key,
    load_spectra_cache,
    load_wavelength,
    save_spectra_cache,
    unit_norm,
)
from external_data.Context.readData import ResultType, json_from_file

# IncInAverage — runs included in calibration average (Chameleon README)
_VALID_RUN_TYPE = ResultType.IncInAverage.value


def _get_ccd_wavelength_intensity(
    analysis: dict,
    run_id: int,
    integration_phase: int,
    ccd_range: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Wavelength + intensity for one CCD range (native pixel grid, no upsampling)."""
    run = None
    for r in analysis["results"]:
        if r["runId"] == run_id:
            run = r
            break
    if run is None:
        raise ValueError(f"run_id {run_id} not found in analysis results")

    ccd_data = None
    for phase, cd in enumerate(run["spectraData"]):
        if phase + 1 == integration_phase:
            ccd_data = cd
            break
    if ccd_data is None:
        raise ValueError(f"integration_phase {integration_phase} not found")

    rng = ccd_data["spectra"][ccd_range - 1]
    intensities = np.asarray(rng["results"], dtype=np.float64)
    drift = [rng["drift"]["beta"], rng["drift"]["alpha"]]
    p2w = rng["pixelToWaveLength"][::-1]
    wavelength = compute_ccd_wavelengths(intensities.size, p2w, drift)
    return wavelength, intensities


def find_first_valid_run(analysis: dict) -> int | None:
    """First run with runId > 0 and type == IncInAverage (2)."""
    for run in analysis["results"]:
        if run["runId"] > 0 and run["type"] == _VALID_RUN_TYPE:
            return int(run["runId"])
    return None


def load_spectrum_from_json(
    json_path: str | Path,
    run_id: int,
    integration_phase: int = 1,
    ccd_ranges: tuple[int, ...] = (1, 2),
    reference_wavelength: np.ndarray | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """
    Load one measured spectrum from a VASKUT-style analysis JSON.

    Concatenates the requested CCD ranges, optionally interpolates onto a
    reference wavelength grid, then applies unit_norm when normalize=True.
    """
    analysis = json_from_file(str(json_path))["analysis"]
    wl_parts: list[np.ndarray] = []
    i_parts: list[np.ndarray] = []
    for ccd in ccd_ranges:
        wl, intensities = _get_ccd_wavelength_intensity(
            analysis, run_id, integration_phase, ccd,
        )
        wl_parts.append(wl)
        i_parts.append(intensities)

    wl_full = np.concatenate(wl_parts)
    i_full = np.concatenate(i_parts)

    if reference_wavelength is not None:
        spectrum = np.interp(
            reference_wavelength, wl_full, i_full, left=0.0, right=0.0,
        )
    else:
        spectrum = i_full.copy()

    if normalize:
        spectrum = unit_norm(spectrum)
    return spectrum.astype(np.float64)


def _load_matrix_concentrations(
    xlsx_path: str,
    db_path: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Return matrix indexed by sample name and DB-filtered element columns."""
    conc = pd.read_excel(xlsx_path, sheet_name="Concentrations")
    conc.columns = [str(c).strip() for c in conc.columns]
    name_col = conc.columns[0]
    elem_cols = [
        c for c in conc.columns[1:]
        if c and not c.lower().startswith("unnamed:")
    ]
    db_elems = _db_elements(db_path)
    elem_cols = [e for e in elem_cols if e in db_elems]
    matrix = conc.set_index(name_col)
    return matrix, elem_cols


def _normalize_concentration_row(row: pd.Series, elem_cols: list[str]) -> np.ndarray:
    """Convert wt% row to fractions summing to 1.0."""
    vals = pd.to_numeric(row[elem_cols], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    total = vals.sum()
    if total <= 0:
        return vals
    if total > 1.5:
        vals = vals / 100.0
        total = vals.sum()
    if total > 0:
        vals = vals / total
    return vals


def discover_measured_json_files(json_root: Path) -> list[Path]:
    """All analysis JSON files under json_root, excluding *_info.json."""
    return sorted(
        p for p in json_root.rglob("*.json")
        if not p.name.endswith("_info.json")
    )


def _instrument_from_path(json_path: Path) -> str:
    """REMUS-xxxx folder name two levels above the JSON file."""
    return json_path.parent.parent.name


def build_measured_entries(
    json_root: str | Path,
    xlsx_path: str,
    db_path: str,
    integration_phase: int = 1,
    run_policy: str = "first_valid",
    seed: int = 42,
    max_files: int | None = None,
    verbose: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    Walk JSON tree and build one metadata row per matched file.

    Returns (entries, stats) where each entry has keys needed to load a spectrum
    plus sample_table columns (without Te/Ne set yet).
    """
    json_root = Path(json_root).expanduser().resolve()
    matrix, elem_cols = _load_matrix_concentrations(xlsx_path, db_path)

    json_files = discover_measured_json_files(json_root)

    name_to_row_idx: dict[str, int] = {}
    for row_idx, name in enumerate(matrix.index, start=1):
        if pd.notna(name):
            name_to_row_idx[str(name).strip()] = row_idx

    entries: list[dict[str, Any]] = []
    stats = {
        "json_total": len(json_files),
        "matched": 0,
        "skipped_no_matrix": 0,
        "skipped_no_valid_run": 0,
        "skipped_error": 0,
    }

    for json_path in json_files:
        sample_name = json_path.stem
        if sample_name not in matrix.index:
            stats["skipped_no_matrix"] += 1
            continue

        try:
            analysis = json_from_file(str(json_path))["analysis"]
            if run_policy == "first_valid":
                run_id = find_first_valid_run(analysis)
            else:
                raise ValueError(f"Unknown run_policy: {run_policy}")
            if run_id is None:
                stats["skipped_no_valid_run"] += 1
                continue

            row_idx = name_to_row_idx.get(sample_name, 0)
            sample_id = _normalize_sample_id(sample_name, row_idx)
            instrument = _instrument_from_path(json_path)
            inst_norm = re.sub(r"[^A-Za-z0-9]+", "_", instrument).strip("_").upper()
            unique_id = f"{sample_id}_{inst_norm}_R{run_id:02d}"

            conc = _normalize_concentration_row(matrix.loc[sample_name], elem_cols)
            entry: dict[str, Any] = {
                "json_path": str(json_path),
                "run_id": run_id,
                "integration_phase": integration_phase,
                "sample_type_id": sample_id,
                "sample_type_name": sample_name,
                "unique_id": unique_id,
                "instrument": instrument,
            }
            for j, elem in enumerate(elem_cols):
                entry[elem] = float(conc[j])
            entries.append(entry)
            stats["matched"] += 1
            if max_files is not None and stats["matched"] >= max_files:
                break
        except Exception as exc:
            stats["skipped_error"] += 1
            if verbose:
                print(f"   WARN: skipping {json_path}: {exc}")

        if max_files is not None and stats["matched"] >= max_files:
            break

    # Deterministic ordering for reproducible cache keys
    entries.sort(key=lambda e: (e["unique_id"], e["json_path"]))
    if verbose:
        print(
            f"Measured JSON scan: {stats['matched']} matched, "
            f"{stats['skipped_no_matrix']} no matrix row, "
            f"{stats['skipped_no_valid_run']} no valid run, "
            f"{stats['skipped_error']} errors"
        )
    return entries, stats


def entries_to_sample_table(
    entries: list[dict[str, Any]],
    elem_cols: list[str],
) -> pd.DataFrame:
    """Build sample_table DataFrame from measured entries."""
    rows = []
    meta_keys = {"json_path", "run_id", "integration_phase", "instrument"}
    for e in entries:
        row = {k: v for k, v in e.items() if k not in meta_keys}
        row["Te"] = 0.0
        row["Ne"] = 0.0
        rows.append(row)

    cols = ["sample_type_id", "sample_type_name", "unique_id"] + elem_cols + ["Te", "Ne"]
    table = pd.DataFrame(rows)
    return table[cols].fillna(0.0)


def load_measured_spectra(
    entries: list[dict[str, Any]],
    reference_wavelength: np.ndarray,
    ccd_ranges: tuple[int, ...] = (1, 2),
    normalize: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    """Load and stack spectra for all measured entries."""
    spectra = np.zeros((len(entries), len(reference_wavelength)), dtype=np.float64)
    for i, entry in enumerate(entries):
        spectra[i] = load_spectrum_from_json(
            entry["json_path"],
            run_id=entry["run_id"],
            integration_phase=entry["integration_phase"],
            ccd_ranges=ccd_ranges,
            reference_wavelength=reference_wavelength,
            normalize=normalize,
        )
        if verbose and (i + 1) % 500 == 0:
            print(f"   Loaded {i + 1}/{len(entries)} spectra")
    if verbose:
        print(f"Loaded {len(entries)} measured spectra.")
    return spectra


class MeasuredLIBSDataset(Dataset):
    """PyTorch dataset for experimental LIBS spectra cached to HDF5."""

    def __init__(
        self,
        json_root: str,
        xlsx_path: str,
        db_path: str,
        wavelength: np.ndarray,
        integration_phase: int = 1,
        ccd_ranges: tuple[int, ...] = (1, 2),
        run_policy: str = "first_valid",
        normalize: bool = True,
        cache_dir: str | None = None,
        seed: int = 42,
        max_files: int | None = None,
        verbose: bool = True,
    ):
        self.json_root = str(Path(json_root).expanduser().resolve())
        self.xlsx_path = str(Path(xlsx_path).expanduser().resolve())
        self.db_path = str(Path(db_path).expanduser().resolve())
        self.wavelength = wavelength
        self.integration_phase = integration_phase
        self.ccd_ranges = ccd_ranges
        self.run_policy = run_policy
        self.normalize = normalize
        self.seed = seed
        self.max_files = max_files
        self.verbose = verbose
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(__file__), "..", "external_data", "cache",
        )
        os.makedirs(self.cache_dir, exist_ok=True)

        self._entries: list[dict[str, Any]] = []
        self.sample_table, self.spectra = self._build()

    @property
    def cache_key(self) -> str:
        cfg = {
            "source": "measured",
            "json_root": self.json_root,
            "xlsx_path": self.xlsx_path,
            "integration_phase": self.integration_phase,
            "ccd_ranges": list(self.ccd_ranges),
            "run_policy": self.run_policy,
            "normalize": self.normalize,
            "n_wavelength": int(self.wavelength.size),
            "wavelength_first": float(self.wavelength[0]),
            "wavelength_last": float(self.wavelength[-1]),
            "seed": self.seed,
            "max_files": self.max_files,
            # The DB decides which element columns the cached sample table keeps.
            **line_db_cache_key(self.db_path),
        }
        return hashlib.md5(
            json.dumps(cfg, sort_keys=True, default=str).encode(),
        ).hexdigest()[:12]

    def _cache_path(self) -> str:
        return os.path.join(self.cache_dir, f"measured_cache_{self.cache_key}.h5")

    def _build(self) -> tuple[pd.DataFrame, np.ndarray]:
        cache = self._cache_path()
        if os.path.isfile(cache):
            if self.verbose:
                print(f"Loading cached measured spectra from: {cache}")
            return load_spectra_cache(cache)

        entries, stats = build_measured_entries(
            json_root=self.json_root,
            xlsx_path=self.xlsx_path,
            db_path=self.db_path,
            integration_phase=self.integration_phase,
            run_policy=self.run_policy,
            seed=self.seed,
            max_files=self.max_files,
            verbose=self.verbose,
        )
        self._entries = entries

        if not entries:
            return pd.DataFrame(), np.empty((0, len(self.wavelength)))

        _, elem_cols = _load_matrix_concentrations(self.xlsx_path, self.db_path)
        table = entries_to_sample_table(entries, elem_cols)

        if self.verbose:
            print(f"\nTotal measured shots to load: {len(entries)}")

        spectra = load_measured_spectra(
            entries,
            reference_wavelength=self.wavelength,
            ccd_ranges=self.ccd_ranges,
            normalize=self.normalize,
            verbose=self.verbose,
        )

        save_spectra_cache(table, spectra, cache)
        if self.verbose:
            print(f"Cached to: {cache}")
        return table, spectra

    def __len__(self) -> int:
        return len(self.sample_table)

    def __getitem__(self, idx: int) -> dict:
        row = self.sample_table.iloc[idx].to_dict()
        row["spectrum"] = self.spectra[idx]
        return row


def build_measured_dataset_from_config(cfg: dict) -> MeasuredLIBSDataset:
    """Top-level entry for measured data (config/libs_data_measured.yaml)."""
    paths = cfg["paths"]
    measured = cfg.get("measured", {})

    db_path = str(Path(paths["db"]).expanduser().resolve())
    xlsx_path = str(Path(paths["sample_matrix"]).expanduser().resolve())
    json_root = str(Path(paths["measured_json_root"]).expanduser().resolve())
    wl_json = str(Path(paths["wavelength_json"]).expanduser().resolve())
    cache_dir = str(Path(paths.get("cache_dir", "external_data/cache")).expanduser().resolve())

    wavelength = load_wavelength(wl_json)
    ccd_ranges = tuple(measured.get("ccd_ranges", [1, 2]))

    return MeasuredLIBSDataset(
        json_root=json_root,
        xlsx_path=xlsx_path,
        db_path=db_path,
        wavelength=wavelength,
        integration_phase=measured.get("integration_phase", 1),
        ccd_ranges=ccd_ranges,
        run_policy=measured.get("run_policy", "first_valid"),
        normalize=measured.get("normalize", "unit_norm") == "unit_norm",
        cache_dir=cache_dir,
        seed=measured.get("seed", 42),
        max_files=measured.get("max_files"),
        verbose=measured.get("verbose", True),
    )
