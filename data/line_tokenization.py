"""
Line-token assembly: combine theoretical dictionary + per-spectrum Voigt fits
into a single HDF5 cache that is directly consumable by a Linear-projection
embedding at training time.

Layout per token (n_features = 14, all float32, RAW values):

    0  central_wavelength            (nm)
    1  Ei                            (eV)
    2  Ek                            (eV)
    3  log10_gi
    4  log10_gk
    5  log10_Ak
    6  log10_theoretical_intensity   (Te×Ne grid maximum from the dictionary)
    7  atomic_number                 (integer Z, stored as float)
    8  ion_binary                    (0 = atomic / "I", 1 = ionic / "II"+)
    9  max_intensity                 (Voigt fit amplitude)
   10  fwhm                          (nm)
   11  r2                            (Voigt fit R²)
   12  delta_lambda                  (fitted centre − theoretical centre, nm)
   13  rmse                          (Voigt fit RMSE)

Normalization mean/std are stored as attrs so the model can z-score at load
time. `fit_valid` is kept in a separate uint8 dataset for the encoder's
`key_padding_mask`.

File format
-----------
external_data/cache/line_tokens_<hash>.h5
├── attrs/  n_spectra, n_lines, n_features, feature_names (JSON),
│           line_dict_hash, line_features_hash, config_hash,
│           feature_mean, feature_std (length n_features)
├── tokens               [n_spectra, n_lines, n_features]   float32, gzip
├── fit_valid            [n_spectra, n_lines]               uint8
└── central_wavelength   [n_lines]                          float32
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np

from data.line_features import (
    FEAT_DELTA_LAM,
    FEAT_FWHM,
    FEAT_MAX_INT,
    FEAT_R2,
    FEAT_RMSE,
    FEAT_VALID,
)

# ─────────────────────────── feature layout ──────────────────────────────

FEATURE_NAMES = [
    "central_wavelength",
    "Ei",
    "Ek",
    "log10_gi",
    "log10_gk",
    "log10_Ak",
    "log10_theoretical_intensity",
    "atomic_number",
    "ion_binary",
    "max_intensity",
    "fwhm",
    "r2",
    "delta_lambda",
    "rmse",
]
N_FEATURES = len(FEATURE_NAMES)

# Convenience indices (model side uses these too).
F_WAVELENGTH = 0
F_EI = 1
F_EK = 2
F_LOG_GI = 3
F_LOG_GK = 4
F_LOG_AK = 5
F_LOG_ITH = 6
F_Z = 7
F_ION = 8
F_MAX_I = 9
F_FWHM = 10
F_R2 = 11
F_DELTA = 12
F_RMSE = 13

# Channels considered "dynamic" (depend on the spectrum) — useful for masking.
DYNAMIC_FEATURE_INDICES = (F_MAX_I, F_FWHM, F_R2, F_DELTA, F_RMSE)
# Default MIP targets: amplitude + FWHM.
MIP_TARGET_INDICES = (F_MAX_I, F_FWHM)


# ─────────────────────────── periodic table ──────────────────────────────
# Z 1–103. Stored verbatim so users can sanity-check by symbol → integer.

_ATOMIC_NUMBERS: dict[str, int] = {
    sym: z + 1
    for z, sym in enumerate(
        # H to Lr (covers every element in LIBS_data.db)
        "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe "
        "Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In "
        "Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf "
        "Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am "
        "Cm Bk Cf Es Fm Md No Lr".split()
    )
}


def atomic_number(symbol: str) -> int:
    """Return Z for an element symbol; raises if unknown."""
    sym = symbol.strip()
    if sym not in _ATOMIC_NUMBERS:
        raise KeyError(f"Unknown element symbol '{symbol}' (not in built-in PT)")
    return _ATOMIC_NUMBERS[sym]


def ion_binary(ion_state: str) -> int:
    """Map ion-state string to 0 (atomic / 'I') or 1 (ionic / 'II'+)."""
    return 0 if str(ion_state).strip().upper() == "I" else 1


# ───────────────────────────── helpers ───────────────────────────────────


def _config_hash(parts: dict[str, Any]) -> str:
    return hashlib.md5(
        json.dumps(parts, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


def _safe_log10(x: np.ndarray, floor: float = 1e-80) -> np.ndarray:
    """log10 with a tiny floor — only catches non-positive values, leaves the
    real LIBS dynamic range (≈ 1e-40 … 1e+8) untouched."""
    return np.log10(np.maximum(x.astype(np.float64), floor)).astype(np.float32)


# ───────────────────────────── builder ───────────────────────────────────


def build_line_tokens_cache(
    line_dict_path: str,
    line_features_path: str,
    cache_dir: Optional[str] = None,
    verbose: bool = True,
) -> str:
    """
    Build (or re-use) the combined per-spectrum token HDF5 cache.

    Args:
        line_dict_path: Path to ``line_dict_*.h5`` produced by
            ``data.line_dictionary.build_line_dictionary``.
        line_features_path: Path to ``line_features_*.h5`` produced by
            ``data.line_features.build_line_features_cache``.
        cache_dir: Where to write the result. Defaults to the parent of
            ``line_features_path``.
        verbose: Print progress messages.

    Returns:
        Path to the resulting ``line_tokens_<hash>.h5`` file.
    """
    line_dict_path = str(line_dict_path)
    line_features_path = str(line_features_path)
    cache_dir_path = (
        Path(cache_dir)
        if cache_dir is not None
        else Path(line_features_path).resolve().parent
    )
    cache_dir_path.mkdir(parents=True, exist_ok=True)

    with h5py.File(line_dict_path, "r") as f:
        dict_hash = str(f.attrs.get("config_hash", ""))
        n_lines_dict = int(f.attrs.get("n_lines", f["central_wavelength"].shape[0]))
        wl = f["central_wavelength"][:].astype(np.float32)
        Ei = f["Ei"][:].astype(np.float32)
        Ek = f["Ek"][:].astype(np.float32)
        gi = f["gi"][:]
        gk = f["gk"][:]
        Ak = f["Ak"][:]
        Ith = f["theoretical_intensity"][:]
        elements = f["vocab"]["element"][:]
        ion_states = f["vocab"]["ion_state"][:]

    with h5py.File(line_features_path, "r") as f:
        feat_hash = str(f.attrs.get("config_hash", ""))
        n_spectra = int(f.attrs["n_spectra"])
        n_lines_feat = int(f.attrs["n_lines"])

    if n_lines_dict != n_lines_feat:
        raise ValueError(
            f"Line count mismatch: dict has {n_lines_dict}, features have {n_lines_feat}"
        )
    n_lines = n_lines_dict

    cfg_hash = _config_hash(
        {
            "line_dict_hash": dict_hash,
            "line_features_hash": feat_hash,
            "feature_names": FEATURE_NAMES,
            "version": 1,
        }
    )
    out_path = cache_dir_path / f"line_tokens_{cfg_hash}.h5"
    if out_path.is_file():
        if verbose:
            print(f"Line tokens cache hit: {out_path}")
        return str(out_path)

    if verbose:
        print(
            f"Building line tokens cache: {n_spectra} spectra × {n_lines} lines "
            f"× {N_FEATURES} features → {out_path}"
        )

    # ── static columns (broadcast across all spectra) ─────────────────────
    elements_str = np.asarray(
        [s.decode() if isinstance(s, (bytes, bytearray)) else str(s) for s in elements]
    )
    ion_states_str = np.asarray(
        [s.decode() if isinstance(s, (bytes, bytearray)) else str(s) for s in ion_states]
    )
    z_per_line = np.asarray(
        [atomic_number(sym) for sym in elements_str], dtype=np.float32
    )
    ion_per_line = np.asarray(
        [ion_binary(s) for s in ion_states_str], dtype=np.float32
    )

    static = np.zeros((n_lines, N_FEATURES), dtype=np.float32)
    static[:, F_WAVELENGTH] = wl
    static[:, F_EI] = Ei
    static[:, F_EK] = Ek
    static[:, F_LOG_GI] = _safe_log10(gi)
    static[:, F_LOG_GK] = _safe_log10(gk)
    static[:, F_LOG_AK] = _safe_log10(Ak)
    static[:, F_LOG_ITH] = _safe_log10(Ith)
    static[:, F_Z] = z_per_line
    static[:, F_ION] = ion_per_line

    # ── write tokens row-by-row to keep peak RAM low ──────────────────────
    with h5py.File(line_features_path, "r") as f_in, h5py.File(out_path, "w") as f_out:
        feats = f_in["features"]  # [n_spectra, n_lines, 6] float32

        f_out.attrs["n_spectra"] = n_spectra
        f_out.attrs["n_lines"] = n_lines
        f_out.attrs["n_features"] = N_FEATURES
        f_out.attrs["feature_names"] = json.dumps(FEATURE_NAMES)
        f_out.attrs["line_dict_hash"] = dict_hash
        f_out.attrs["line_features_hash"] = feat_hash
        f_out.attrs["line_dict_path"] = line_dict_path
        f_out.attrs["line_features_path"] = line_features_path
        f_out.attrs["config_hash"] = cfg_hash
        f_out.attrs["dynamic_feature_indices"] = list(DYNAMIC_FEATURE_INDICES)
        f_out.attrs["mip_target_indices"] = list(MIP_TARGET_INDICES)

        tokens_ds = f_out.create_dataset(
            "tokens",
            shape=(n_spectra, n_lines, N_FEATURES),
            dtype=np.float32,
            chunks=(1, n_lines, N_FEATURES),
            compression="gzip",
            compression_opts=4,
        )
        valid_ds = f_out.create_dataset(
            "fit_valid",
            shape=(n_spectra, n_lines),
            dtype=np.uint8,
            chunks=(1, n_lines),
            compression="gzip",
            compression_opts=4,
        )
        f_out.create_dataset("central_wavelength", data=wl)

        # Running stats (Welford), valid-fit-only for dynamic channels.
        sum_all = np.zeros(N_FEATURES, dtype=np.float64)
        sum_sq_all = np.zeros(N_FEATURES, dtype=np.float64)
        count_all = np.zeros(N_FEATURES, dtype=np.float64)

        # Process in chunks to keep memory bounded.
        chunk = max(1, min(256, n_spectra))
        for start in range(0, n_spectra, chunk):
            stop = min(start + chunk, n_spectra)
            fit_chunk = feats[start:stop]  # [c, n_lines, 6]
            valid = (fit_chunk[..., FEAT_VALID] > 0.5).astype(np.uint8)

            row = np.broadcast_to(
                static[None, :, :], (stop - start, n_lines, N_FEATURES)
            ).copy()
            row[:, :, F_MAX_I] = fit_chunk[..., FEAT_MAX_INT]
            row[:, :, F_FWHM] = fit_chunk[..., FEAT_FWHM]
            row[:, :, F_R2] = fit_chunk[..., FEAT_R2]
            row[:, :, F_DELTA] = fit_chunk[..., FEAT_DELTA_LAM]
            row[:, :, F_RMSE] = fit_chunk[..., FEAT_RMSE]

            tokens_ds[start:stop] = row
            valid_ds[start:stop] = valid

            # Stats: static channels contribute every row; dynamic channels
            # contribute only where the Voigt fit succeeded.
            valid_mask = valid.astype(bool)
            for ci in range(N_FEATURES):
                if ci in DYNAMIC_FEATURE_INDICES:
                    vals = row[:, :, ci][valid_mask]
                else:
                    vals = row[:, :, ci].reshape(-1)
                if vals.size:
                    sum_all[ci] += vals.sum(dtype=np.float64)
                    sum_sq_all[ci] += (vals.astype(np.float64) ** 2).sum()
                    count_all[ci] += vals.size

            if verbose and (start // chunk) % max(1, n_spectra // (chunk * 20)) == 0:
                print(f"  {stop}/{n_spectra} spectra tokenized")

        # Finalize stats.
        mean = np.zeros(N_FEATURES, dtype=np.float32)
        std = np.ones(N_FEATURES, dtype=np.float32)
        for ci in range(N_FEATURES):
            n = max(count_all[ci], 1.0)
            mu = sum_all[ci] / n
            var = max(sum_sq_all[ci] / n - mu * mu, 0.0)
            mean[ci] = float(mu)
            sigma = float(np.sqrt(var))
            std[ci] = sigma if sigma > 1e-8 else 1.0

        f_out.attrs["feature_mean"] = mean
        f_out.attrs["feature_std"] = std

    if verbose:
        print(f"Saved line tokens: {out_path}")
        print("  feature stats (mean / std):")
        for name, m, s in zip(FEATURE_NAMES, mean, std):
            print(f"    {name:<32s} {m:+.3e}  /  {s:.3e}")

    return str(out_path)


# ──────────────────────────── reader ─────────────────────────────────────


class LineTokensStore:
    """Lazy HDF5 reader for ``line_tokens_<hash>.h5``."""

    def __init__(self, path: str):
        self.path = str(path)
        self._file: h5py.File | None = None

    def _ensure_open(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file

    # ── metadata ─────────────────────────────────────────────────────────

    @property
    def n_spectra(self) -> int:
        return int(self._ensure_open().attrs["n_spectra"])

    @property
    def n_lines(self) -> int:
        return int(self._ensure_open().attrs["n_lines"])

    @property
    def n_features(self) -> int:
        return int(self._ensure_open().attrs["n_features"])

    @property
    def feature_names(self) -> list[str]:
        return json.loads(self._ensure_open().attrs["feature_names"])

    @property
    def feature_mean(self) -> np.ndarray:
        return np.asarray(self._ensure_open().attrs["feature_mean"], dtype=np.float32)

    @property
    def feature_std(self) -> np.ndarray:
        return np.asarray(self._ensure_open().attrs["feature_std"], dtype=np.float32)

    @property
    def central_wavelength(self) -> np.ndarray:
        return self._ensure_open()["central_wavelength"][:].astype(np.float32)

    # ── per-spectrum access ──────────────────────────────────────────────

    def get(self, spectrum_idx: int) -> tuple[np.ndarray, np.ndarray]:
        f = self._ensure_open()
        return (
            f["tokens"][spectrum_idx].astype(np.float32),
            f["fit_valid"][spectrum_idx].astype(np.uint8),
        )

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Tokenize per-spectrum line features by combining the theoretical "
            "dictionary with the Voigt fits into a single HDF5 cache that can "
            "be reused for training with a nn.Linear embedding."
        )
    )
    parser.add_argument("--line_dict_path", type=str, required=True)
    parser.add_argument("--line_features_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None)
    args = parser.parse_args()

    out = build_line_tokens_cache(
        line_dict_path=args.line_dict_path,
        line_features_path=args.line_features_path,
        cache_dir=args.cache_dir,
        verbose=True,
    )
    print(out)
