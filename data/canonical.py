"""
Canonical spectral representation shared by real maps and synthetic spectra.

Real LIBS spectra carry nuisance the synthetic generator does not model: per-channel
read/shot noise (Mar1A rock: 3.2 / 9.5 / 48 counts in the UV / blue / red channel), a
signal-dependent continuum (red channel ~230-450 counts), exact zeros after dark
subtraction, and the scattered 266 nm laser line. None of it carries mineral
information (a real-label probe scores the same with and without it), but because only
the real domain has it, a model trained on synthetic spectra meets it out of
distribution and uses it as a "realness" shortcut. Instead of simulating it,
``canonicalise`` removes it from both domains identically:

    x in full-well units (counts / full_well_counts) [n, n_px]
      -> per detector channel (seams where the axis stops increasing):
         baseline over ``baseline_nm``: 'runmin' = moving mean of a running minimum (the
         reference of the linear screen; it sits ~2.5 sigma below the noise mean, so noise
         peaks above ~mean + 0.5 sigma survive the floor) or 'median' = block medians
         interpolated per pixel (sits at the noise mean; the floor then removes ~99.9 % of
         pure noise and the detection limit is the same in both domains)
      -> soft noise floor: max(x - baseline - k_sigma * sigma_c, 0), sigma_c in full-well units
      -> scattered laser line blanked (``blank_nm``)
      -> saturation mask: x >= ``saturation_fw`` (absolute, per pixel)
      -> L1 normalisation over unsaturated pixels (mean pixel ~ 1)

The noise floor removes the weak lines that are below the real detection limit; for it to
mean the same in both domains, synthetic spectra must keep their absolute scale
(``generation.detector.output_units: full_well``, data/two_zone_pipeline.py).

``patch_input`` is the single builder of spectral_patch model input (training, map
inference, evaluation):

    preprocess 'none'      -> unit_norm rows (the historic input)        [n, n_px]
    preprocess 'canonical' -> [canonical spectrum, saturation mask]     [n, 2 n_px]

Units: wavelengths in nm; ``units`` is 'unit_norm' (unit-normalised cache), 'full_well'
(fraction of full well) or 'counts' (raw detector counts).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.ndimage import minimum_filter1d, uniform_filter1d

from data.efficiency_correction import channel_segments

PREPROCESS_MODES = ("none", "canonical")
UNITS = ("unit_norm", "full_well", "counts")
DEFAULT_CANONICAL: dict[str, Any] = {
    "full_well_counts": 64900.0,  # Avantes/FireFly plateau (Mar1A: 64,850-64,950 counts)
    "noise_sigma_counts": [3.2, 9.5, 48.2],  # per channel; Mar1A rock, robust first-difference MAD
    "k_sigma": 3.0,  # soft threshold at k_sigma * sigma
    "baseline_nm": 2.0,  # window of the baseline estimate
    "baseline_method": "runmin",  # runmin | median (see module docstring)
    "blank_nm": [[265.4, 267.0]],  # scattered 266 nm laser line
    "saturation_fw": 0.99,  # pixels at >= 99 % of full well count as clipped
}


def unit_norm_rows(X: np.ndarray) -> np.ndarray:
    """Row-wise ``libs_pipeline.unit_norm`` (min-shift, divide by the maximum), float32."""
    X = np.asarray(X, dtype=np.float32)
    X = X - X.min(axis=1, keepdims=True)
    peak = X.max(axis=1, keepdims=True)
    return np.divide(X, peak, out=np.zeros_like(X), where=peak > 0)


def channel_bounds(wavelength: np.ndarray) -> list[tuple[int, int]]:
    """Pixel ranges of the detector channels (new channel where the axis stops increasing)."""
    seg = channel_segments(np.asarray(wavelength, dtype=np.float64))
    edges = np.flatnonzero(np.diff(seg)) + 1
    starts = np.r_[0, edges]
    ends = np.r_[edges, seg.size]
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def _block_median_baseline(seg: np.ndarray, win: int) -> np.ndarray:
    """Median of consecutive ``win``-pixel blocks, linearly interpolated to every pixel
    (fast O(n) robust baseline; [n, m] -> [n, m])."""
    n, m = seg.shape
    nb = max(1, m // win)
    edges = np.linspace(0, m, nb + 1).astype(int)
    med = np.stack([np.median(seg[:, a:b], axis=1) for a, b in zip(edges[:-1], edges[1:])], 1)
    centres = 0.5 * (edges[:-1] + edges[1:] - 1)
    if nb == 1:
        return np.repeat(med, m, axis=1)
    px = np.arange(m)
    j = np.clip(np.searchsorted(centres, px) - 1, 0, nb - 2)
    w = np.clip((px - centres[j]) / (centres[j + 1] - centres[j]), 0.0, 1.0)
    return med[:, j] * (1.0 - w) + med[:, j + 1] * w


def canonical_config(cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """``cfg`` merged over DEFAULT_CANONICAL."""
    out = {k: (list(v) if isinstance(v, list) else v) for k, v in DEFAULT_CANONICAL.items()}
    out.update(dict(cfg or {}))
    return out


def canonicalise(
    x_fw: np.ndarray, wavelength: np.ndarray, cfg: Mapping[str, Any] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Canonical spectra and saturation mask of spectra in full-well units.

    Args:
        x_fw:       [n, n_px] or [n_px] spectra in full-well units (counts / full_well_counts)
        wavelength: [n_px] spectrometer axis in nm (may be non-monotonic at channel seams)
        cfg:        overrides of DEFAULT_CANONICAL
    Returns:
        canon [n, n_px] float32 (L1-normalised over unsaturated pixels, mean pixel ~ 1),
        sat   [n, n_px] bool (clipped pixels)
    """
    c = canonical_config(cfg)
    x = np.atleast_2d(np.asarray(x_fw, dtype=np.float32))
    wl = np.asarray(wavelength, dtype=np.float64).reshape(-1)
    if x.shape[1] != wl.size:
        raise ValueError(f"spectra have {x.shape[1]} px, the axis {wl.size}")
    bounds = channel_bounds(wl)
    sigma = np.atleast_1d(np.asarray(c["noise_sigma_counts"], dtype=np.float64))
    if sigma.size == 1:
        sigma = np.repeat(sigma, len(bounds))
    if sigma.size != len(bounds):
        raise ValueError(
            f"noise_sigma_counts has {sigma.size} values, the axis has {len(bounds)} channels"
        )
    floors = float(c["k_sigma"]) * sigma / float(c["full_well_counts"])

    out = np.empty_like(x)
    for k, (a, b) in enumerate(bounds):
        seg = x[:, a:b]
        px_nm = float(np.median(np.abs(np.diff(wl[a:b])))) if b - a > 1 else 1.0
        win = max(5, int(round(float(c["baseline_nm"]) / max(px_nm, 1e-9))))
        win = min(win, max(1, b - a))
        if c["baseline_method"] == "median":
            base = _block_median_baseline(seg, win)
        elif c["baseline_method"] == "runmin":
            base = uniform_filter1d(minimum_filter1d(seg, win, axis=1), win, axis=1)
        else:
            raise ValueError(f"baseline_method must be runmin|median, got {c['baseline_method']!r}")
        out[:, a:b] = np.maximum(seg - base - floors[k], 0.0)
    blank = np.zeros(wl.size, dtype=bool)
    for lo, hi in c["blank_nm"] or ():
        blank |= (wl > float(lo)) & (wl < float(hi))
    out[:, blank] = 0.0
    sat = x >= float(c["saturation_fw"])
    total = np.where(sat, 0.0, out).sum(axis=1, keepdims=True)
    total[total <= 0] = 1.0
    out = (out / total * out.shape[1]).astype(np.float32)
    return out, sat


def resolve_spec(
    model_cfg: Mapping[str, Any] | None, libs_cfg: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Input spec {preprocess, canonical} of a spectral_patch model: ``model.patch.preprocess``
    and ``model.patch.canonical``, with ``full_well_counts`` from the data config's
    ``generation.detector`` when present."""
    patch = dict((model_cfg or {}).get("patch") or {})
    mode = str(patch.get("preprocess", "none"))
    if mode not in PREPROCESS_MODES:
        raise ValueError(f"model.patch.preprocess must be one of {PREPROCESS_MODES}, got {mode!r}")
    can = dict(patch.get("canonical") or {})
    det = ((libs_cfg or {}).get("generation") or {}).get("detector") or {}
    if "full_well_counts" in det and "full_well_counts" not in can:
        can["full_well_counts"] = float(det["full_well_counts"])
    return {"preprocess": mode, "canonical": canonical_config(can)}


def require_unit_norm(ds: Any, what: str) -> None:
    """Refuse a dataset whose spectra are not unit-normalised (e.g. a full-well cache) in a
    code path that consumes ``ds.spectra`` directly."""
    units = getattr(ds, "units", "unit_norm")
    if units != "unit_norm":
        raise ValueError(
            f"{what} expects unit-normalised spectra, but the dataset holds {units!r} spectra "
            "(full-well data is meant for the canonical spectral_patch input)"
        )


def patch_input(
    x: np.ndarray, wavelength: np.ndarray, units: str, spec: Mapping[str, Any] | None
) -> np.ndarray:
    """The spectral_patch model input of spectra ``x`` [n, n_px] given in ``units``.

    'none': unit_norm rows [n, n_px] (bit-identical to the unit-normalised caches);
    'canonical': concatenated [canonicalise(x_fw), saturation mask] [n, 2 n_px].
    """
    if units not in UNITS:
        raise ValueError(f"units must be one of {UNITS}, got {units!r}")
    spec = dict(spec or {})
    mode = str(spec.get("preprocess", "none"))
    if mode == "none":
        return unit_norm_rows(x)
    if mode != "canonical":
        raise ValueError(f"unknown preprocess {mode!r}")
    if units == "unit_norm":
        raise ValueError(
            "canonical input needs an absolute scale; unit-normalised spectra lost it "
            "(use generation.detector.output_units: full_well)"
        )
    cfg = canonical_config(spec.get("canonical"))
    x = np.asarray(x, dtype=np.float32)
    x_fw = x / float(cfg["full_well_counts"]) if units == "counts" else x
    canon, sat = canonicalise(x_fw, wavelength, cfg)
    return np.concatenate([canon, sat.astype(np.float32)], axis=1)
