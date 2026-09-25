"""
Relative efficiency correction (REC) of LIGHTIGO / FireFly spectra.

Port of the R procedure ``external_data/Data/20240910_REC_FireFly.R`` (Avantes
ULS4096CL spectrometers behind the FireFly LIBS head).  Two calibration-lamp
measurements define the wavelength-dependent sensitivity of the instrument:

* **deuterium lamp** for the UV part (``uv_range``, default 180-400 nm): the
  measured lamp spectrum is smoothed with a LOWESS filter (``lowess_frac`` of the
  pixels, robust iterations as in R) because the UV signal is weak, and the
  correction is ``certified_irradiance / smoothed_measured``;
* **halogen lamp** for the VIS part (``vis_range``, default 350-900 nm):
  ``certified_irradiance / measured`` without smoothing.

In the overlap (350-400 nm) the deuterium-based curve is fitted to the
halogen-based one with a straight line (``vis = a + b * uv``, R's ``lm``) and
rescaled, so the two lamps of different brightness and exposure join
continuously.  The combined curve is a multiplicative factor
``corrected = measured * factor(wavelength)``; outside the calibrated range it
is undefined (NaN, R's ``rule = 1``).  ``1 / factor`` is the relative
sensitivity of the instrument.

Deviations from the R script (documented and switchable):
* the FireFly axis is made of five spectrometer channels (gain steps near 260,
  370, 460 and 690 nm; the axis runs backwards at 370 and 460 nm).  Smoothing
  and zero-filling are done per channel segment (``channel_edges_nm``) so the
  LOWESS window never straddles a gain step (R smooths across it);
* the halogen spectrum can optionally be smoothed too (``halogen_lowess_frac``,
  off by default as in R; useful where the halogen signal is weak);
* zero / negative pixels (dead pixels, channel seams) are replaced by linear
  interpolation between the nearest valid neighbours in pixel order instead of
  the mean of the next / previous two pixels (which can themselves be zero);
* in the overlap the two curves are cross-faded (``overlap="crossfade"``)
  instead of interleaving their nodes (``overlap="interleave"`` reproduces R).

The certified lamp irradiances are relative (a.u. per nm) - only the shape of
the correction is meaningful, hence *relative* efficiency correction.

Usage::

    from data.efficiency_correction import RelativeEfficiencyCorrection, correct_spectra

    rec = RelativeEfficiencyCorrection.from_lamp_files(
        "external_data/Data/DeuteriumWithCorrection.h5",
        "external_data/Data/HalogenWithCorrection.h5")
    corrected = rec.apply(spectra, wavelength)          # spectra (n, n_px) or (n_px,)
    factor = rec(wavelength)                            # multiplicative factor, NaN outside range
    corrected = correct_spectra(spectra, wavelength, deuterium_h5, halogen_h5)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Certified lamp spectra (relative spectral irradiance vs wavelength [nm]),
# copied from 20240910_REC_FireFly.R
# ─────────────────────────────────────────────────────────────────────────────
UV_LAMP_NM = np.array(
    [
        180,
        220,
        230,
        240,
        250,
        260,
        270,
        280,
        290,
        300,
        310,
        320,
        330,
        340,
        350,
        360,
        370,
        380,
        390,
        400,
    ],
    dtype=np.float64,
)
UV_LAMP_IRRADIANCE = np.array(
    [
        4.14924e1,
        3.1362e1,
        2.9241e1,
        2.6387e1,
        2.3884e1,
        2.1531e1,
        1.9062e1,
        1.6373e1,
        1.4108e1,
        1.2303e1,
        1.098e1,
        9.5562e0,
        8.5976e0,
        7.7735e0,
        7.0041e0,
        6.4071e0,
        5.9195e0,
        5.6728e0,
        5.1471e0,
        5.1181e0,
    ],
    dtype=np.float64,
)
VIS_LAMP_NM = np.array(
    [
        350,
        360,
        370,
        380,
        390,
        400,
        420,
        440,
        460,
        480,
        500,
        525,
        550,
        575,
        600,
        650,
        700,
        750,
        800,
        850,
        900,
        950,
        1000,
        1050,
    ],
    dtype=np.float64,
)
VIS_LAMP_IRRADIANCE = np.array(
    [
        1.9862e-1,
        2.2677e-1,
        2.6293e-1,
        2.9921e-1,
        3.3594e-1,
        3.7371e-1,
        5.0574e-1,
        6.9292e-1,
        9.3803e-1,
        1.2383e0,
        1.5973e0,
        2.0941e0,
        2.6431e0,
        3.261e0,
        3.9116e0,
        5.2424e0,
        6.6042e0,
        7.8885e0,
        9.0153e0,
        1.0003e1,
        1.0732e1,
        1.0926e1,
        1.0511e1,
        9.65e0,
    ],
    dtype=np.float64,
)

DEFAULT_CHANNEL_EDGES_NM = (260.0, 370.0, 460.0, 690.0)  # FireFly: five Avantes channels
DEFAULT_UV_RANGE = (180.0, 400.0)
DEFAULT_VIS_RANGE = (350.0, 900.0)


# ─────────────────────────────────────────────────────────────────────────────
# Small numerical helpers
# ─────────────────────────────────────────────────────────────────────────────
def interp_nan(xq: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Linear interpolation like R's ``approxfun(rule = 1, ties = mean)``:
    ``x`` is sorted, duplicates are averaged, queries outside [min, max] give NaN."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    ux, inv = np.unique(xs, return_inverse=True)
    if ux.size != xs.size:  # ties -> mean
        sums = np.bincount(inv, weights=ys, minlength=ux.size)
        cnts = np.bincount(inv, minlength=ux.size)
        ys = sums / cnts
        xs = ux
    xq = np.asarray(xq, dtype=np.float64)
    out = np.interp(xq, xs, ys, left=np.nan, right=np.nan)
    return out


def fill_nonpositive(y: np.ndarray) -> np.ndarray:
    """Replace ``<= 0`` (or non-finite) samples by linear interpolation between the
    nearest valid neighbours in array order (edges take the nearest valid value)."""
    y = np.asarray(y, dtype=np.float64).copy()
    bad = ~np.isfinite(y) | (y <= 0)
    if not bad.any():
        return y
    good = np.nonzero(~bad)[0]
    if good.size == 0:
        raise ValueError("spectrum has no positive sample")
    idx = np.arange(y.size)
    y[bad] = np.interp(idx[bad], good, y[good])
    return y


def channel_segments(
    wavelength: np.ndarray,
    channel_edges_nm: Sequence[float] | None = None,
    min_segment_px: int = 8,
) -> np.ndarray:
    """Segment id per pixel: a new segment starts where the axis stops increasing
    (spectrometer seam) or where it crosses one of ``channel_edges_nm``.
    Segments shorter than ``min_segment_px`` (e.g. the few pixels between an
    edge and the seam right behind it) are merged into the preceding one."""
    wl = np.asarray(wavelength, dtype=np.float64)
    new = np.zeros(wl.size, dtype=bool)
    if wl.size > 1:
        new[1:] |= np.diff(wl) <= 0
        for edge in channel_edges_nm or ():
            new[1:] |= (wl[:-1] < edge) & (wl[1:] >= edge)
    starts = np.nonzero(new)[0]
    bounds = np.concatenate([[0], starts, [wl.size]])
    keep = [0]
    for b in bounds[1:-1]:
        if b - keep[-1] >= min_segment_px:
            keep.append(int(b))
    new = np.zeros(wl.size, dtype=bool)
    new[keep[1:]] = True
    return np.cumsum(new)


def _per_segment(y: np.ndarray, seg: np.ndarray, fn) -> np.ndarray:
    """Apply ``fn`` (1-D array -> 1-D array) to each contiguous segment of ``y``."""
    out = np.empty_like(np.asarray(y, dtype=np.float64))
    for s_id in np.unique(seg):
        m = seg == s_id
        out[m] = fn(np.asarray(y, dtype=np.float64)[m])
    return out


def lowess(y: np.ndarray, frac: float = 0.01, iters: int = 3) -> np.ndarray:
    """LOWESS smoother over the sample index, following R's ``lowess(y, f, iter)``:
    local linear fits with tricube weights over the ``ceil(frac * n)`` nearest
    samples and ``iters`` robustifying iterations with bisquare weights on the
    residuals (6 x median |residual|).  Returns the smoothed values."""
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if n < 3:
        return y.copy()
    ns = int(min(max(int(frac * n + 1e-7), 2), n))
    half = (ns - 1) // 2
    x = np.arange(n, dtype=np.float64)
    robust = np.ones(n)
    fitted = y.copy()
    for it in range(iters + 1):
        for i in range(n):
            lo = i - half
            hi = lo + ns
            if lo < 0:
                lo, hi = 0, ns
            elif hi > n:
                lo, hi = n - ns, n
            xs = x[lo:hi]
            d = np.abs(xs - x[i])
            h = d.max()
            if h <= 0:
                fitted[i] = y[i]
                continue
            r = d / h
            w = np.where(r < 0.001, 1.0, np.where(r > 0.999, 0.0, (1.0 - r**3) ** 3))
            w = w * robust[lo:hi]
            sw = w.sum()
            if sw <= 0:
                fitted[i] = y[i]
                continue
            ys = y[lo:hi]
            xm = (w * xs).sum() / sw
            ym = (w * ys).sum() / sw
            dx = xs - xm
            sxx = (w * dx * dx).sum()
            if sxx > 1e-12 * (h * h):
                slope = (w * dx * (ys - ym)).sum() / sxx
                fitted[i] = ym + slope * (x[i] - xm)
            else:
                fitted[i] = ym
        if it == iters:
            break
        res = y - fitted
        cmad = 6.0 * np.median(np.abs(res))
        if cmad <= 1e-7 * np.mean(np.abs(res)) + 1e-300:
            break
        u = np.abs(res) / cmad
        robust = np.where(u < 0.001, 1.0, np.where(u > 0.999, 0.0, (1.0 - u**2) ** 2))
    return fitted


# ─────────────────────────────────────────────────────────────────────────────
# LIGHTIGO HDF5 access
# ─────────────────────────────────────────────────────────────────────────────
def load_lightigo_axis(path: str | Path, measurement: str | None = None) -> np.ndarray:
    """Wavelength axis [nm] of a LIGHTIGO HDF5 file (``measurements/<key>/libs/calibration``)."""
    import h5py

    with h5py.File(path, "r") as f:
        meas = f["measurements"]
        key = measurement or sorted(meas)[0]
        return meas[key]["libs"]["calibration"][...].astype(np.float64)


def load_lightigo_spectrum(
    path: str | Path, measurement: str | None = None, spectrum: int | None = None
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Wavelength axis and one spectrum of a LIGHTIGO HDF5 file
    (``measurements/<key>/libs/{calibration, data, metadata}``).

    ``spectrum=None`` returns the mean of all valid shots (the lamp files hold a
    single shot, which is then returned unchanged); an integer selects one shot.
    Returns (wavelength [nm], spectrum [counts], metadata summary)."""
    import h5py

    with h5py.File(path, "r") as f:
        meas = f["measurements"]
        key = measurement or sorted(meas)[0]
        g = meas[key]["libs"]
        wl = g["calibration"][...].astype(np.float64)
        data = g["data"]
        md = (
            {k: np.asarray(g["metadata"][k][...]) for k in g["metadata"]} if "metadata" in g else {}
        )
        if spectrum is None:
            X = data[...].astype(np.float64)
            inv = md.get("Invalid")
            if inv is not None:
                X = X[inv.ravel() == 0]
            spec = X.mean(axis=0)
        else:
            spec = data[int(spectrum)].astype(np.float64)
    info = {
        "file": str(path),
        "measurement": key,
        "n_px": int(wl.size),
        "exposure_us": (
            float(np.ravel(md["Exposure Time us"])[0]) if "Exposure Time us" in md else None
        ),
        "accumulation": int(np.ravel(md["Accumulation"])[0]) if "Accumulation" in md else None,
    }
    return wl, spec, info


# ─────────────────────────────────────────────────────────────────────────────
# The correction
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RelativeEfficiencyCorrection:
    """Multiplicative relative-efficiency correction ``factor(wavelength)``.

    Attributes:
        wavelength:   nodes of the combined correction [nm], sorted and unique
        factor:       correction at the nodes (a.u.); ``measured * factor`` is
                      proportional to the true spectral radiance
        valid_range:  (min, max) wavelength [nm] where the correction is defined
        uv_scale:     (intercept, slope) that maps the deuterium-based curve onto
                      the halogen-based one in the overlap
        uv_nodes, uv_factor:   deuterium-based curve before rescaling (diagnostics)
        vis_nodes, vis_factor: halogen-based curve (diagnostics)
        info:         provenance (files, ranges, options, signal levels)
    """

    wavelength: np.ndarray
    factor: np.ndarray
    valid_range: tuple[float, float]
    uv_scale: tuple[float, float] = (0.0, 1.0)
    uv_nodes: np.ndarray = field(default_factory=lambda: np.empty(0))
    uv_factor: np.ndarray = field(default_factory=lambda: np.empty(0))
    vis_nodes: np.ndarray = field(default_factory=lambda: np.empty(0))
    vis_factor: np.ndarray = field(default_factory=lambda: np.empty(0))
    info: dict[str, Any] = field(default_factory=dict)

    # ---- construction --------------------------------------------------------
    @classmethod
    def from_lamp_files(
        cls,
        deuterium_h5: str | Path,
        halogen_h5: str | Path,
        *,
        measurement: str | None = None,
        spectrum: int | None = None,
        **options: Any,
    ) -> "RelativeEfficiencyCorrection":
        """Build the correction from the two LIGHTIGO lamp measurements.
        Keyword options are those of :meth:`from_arrays`."""
        wl_d, spec_d, info_d = load_lightigo_spectrum(deuterium_h5, measurement, spectrum)
        wl_h, spec_h, info_h = load_lightigo_spectrum(halogen_h5, measurement, spectrum)
        rec = cls.from_arrays(wl_d, spec_d, wl_h, spec_h, **options)
        rec.info.update(deuterium=info_d, halogen=info_h)
        return rec

    @classmethod
    def from_arrays(
        cls,
        wl_deuterium: np.ndarray,
        spec_deuterium: np.ndarray,
        wl_halogen: np.ndarray,
        spec_halogen: np.ndarray,
        *,
        uv_range: tuple[float, float] = DEFAULT_UV_RANGE,
        vis_range: tuple[float, float] = DEFAULT_VIS_RANGE,
        lowess_frac: float = 0.01,
        lowess_iters: int = 3,
        uv_lamp: tuple[np.ndarray, np.ndarray] | None = None,
        vis_lamp: tuple[np.ndarray, np.ndarray] | None = None,
        overlap_fit: str = "linear",
        overlap: str = "crossfade",
        channel_edges_nm: Sequence[float] | None = DEFAULT_CHANNEL_EDGES_NM,
        halogen_lowess_frac: float | None = None,
    ) -> "RelativeEfficiencyCorrection":
        """Core builder (the R procedure) on in-memory arrays.

        Args:
            wl_*, spec_*: wavelength axis [nm] and measured lamp spectrum [counts]
                          in pixel order (the axis need not be monotonic)
            uv_range:     wavelengths taken from the deuterium measurement
            vis_range:    wavelengths taken from the halogen measurement
            lowess_frac:  LOWESS span for the deuterium spectrum (fraction of pixels)
            lowess_iters: robustifying iterations of the LOWESS filter
            uv_lamp, vis_lamp: (wavelength, irradiance) tables overriding the
                          certified values copied from the R script
            overlap_fit:  "linear" (``vis = a + b uv``, R's lm) or "scale" (``vis = b uv``)
            overlap:      "crossfade" (linear blend of the two curves across the
                          overlap) or "interleave" (R: nodes of both curves kept)
            channel_edges_nm: wavelengths of spectrometer gain steps; smoothing and
                          zero-filling never cross them (axis reversals are
                          detected automatically); None = whole range at once
            halogen_lowess_frac: LOWESS span for the halogen spectrum, None = raw (R)
        """
        uv_lamp_nm, uv_lamp_val = (
            uv_lamp if uv_lamp is not None else (UV_LAMP_NM, UV_LAMP_IRRADIANCE)
        )
        vis_lamp_nm, vis_lamp_val = (
            vis_lamp if vis_lamp is not None else (VIS_LAMP_NM, VIS_LAMP_IRRADIANCE)
        )
        if overlap_fit not in ("linear", "scale"):
            raise ValueError("overlap_fit must be 'linear' or 'scale'")
        if overlap not in ("crossfade", "interleave"):
            raise ValueError("overlap must be 'crossfade' or 'interleave'")

        wl_d = np.asarray(wl_deuterium, dtype=np.float64)
        wl_h = np.asarray(wl_halogen, dtype=np.float64)
        spec_d = np.asarray(spec_deuterium, dtype=np.float64)
        spec_h = np.asarray(spec_halogen, dtype=np.float64)

        # ---- deuterium / UV: smoothed measured spectrum vs certified lamp ----
        m_uv = (wl_d > uv_range[0]) & (wl_d < uv_range[1])
        if m_uv.sum() < 10:
            raise ValueError("deuterium axis has fewer than 10 pixels inside uv_range")
        seg_d = channel_segments(wl_d, channel_edges_nm)[m_uv]
        smooth_uv = _per_segment(
            fill_nonpositive(spec_d[m_uv]), seg_d, lambda y: lowess(y, lowess_frac, lowess_iters)
        )
        smooth_uv = fill_nonpositive(smooth_uv)
        uv_nodes = wl_d[m_uv]
        uv_factor = interp_nan(uv_nodes, uv_lamp_nm, uv_lamp_val) / smooth_uv

        # ---- halogen / VIS: raw measured spectrum vs certified lamp -----------
        m_vis = (wl_h > vis_range[0]) & (wl_h < vis_range[1])
        if m_vis.sum() < 10:
            raise ValueError("halogen axis has fewer than 10 pixels inside vis_range")
        seg_h = channel_segments(wl_h, channel_edges_nm)[m_vis]
        meas_vis = fill_nonpositive(spec_h[m_vis])
        if halogen_lowess_frac:
            meas_vis = fill_nonpositive(
                _per_segment(
                    meas_vis, seg_h, lambda y: lowess(y, halogen_lowess_frac, lowess_iters)
                )
            )
        vis_nodes = wl_h[m_vis]
        vis_factor = interp_nan(vis_nodes, vis_lamp_nm, vis_lamp_val) / meas_vis

        # ---- join the two lamps in the overlap --------------------------------
        ov_lo, ov_hi = vis_range[0], uv_range[1]
        if ov_hi <= ov_lo:
            raise ValueError("uv_range and vis_range must overlap")
        wl_ov = np.concatenate(
            [
                uv_nodes[(uv_nodes > ov_lo) & (uv_nodes < ov_hi)],
                vis_nodes[(vis_nodes > ov_lo) & (vis_nodes < ov_hi)],
            ]
        )
        wl_ov = np.unique(wl_ov)
        u = interp_nan(wl_ov, uv_nodes, uv_factor)
        v = interp_nan(wl_ov, vis_nodes, vis_factor)
        ok = np.isfinite(u) & np.isfinite(v)
        if ok.sum() < 3:
            raise ValueError("not enough valid pixels in the UV/VIS overlap to join the lamps")
        if overlap_fit == "linear":
            A = np.column_stack([np.ones(ok.sum()), u[ok]])
            coef, *_ = np.linalg.lstsq(A, v[ok], rcond=None)
            intercept, slope = float(coef[0]), float(coef[1])
        else:
            slope = float((u[ok] @ v[ok]) / (u[ok] @ u[ok]))
            intercept = 0.0
        uv_scaled = intercept + slope * uv_factor

        if overlap == "interleave":
            nodes = np.concatenate([uv_nodes, vis_nodes])
            values = np.concatenate([uv_scaled, vis_factor])
        else:
            # UV-only part, cross-faded overlap, VIS-only part
            uv_only = uv_nodes <= ov_lo
            vis_only = vis_nodes >= ov_hi
            t = (wl_ov - ov_lo) / (ov_hi - ov_lo)  # 0 at the UV side, 1 at the VIS side
            us = intercept + slope * u
            blend = np.where(
                np.isfinite(us) & np.isfinite(v),
                (1.0 - t) * us + t * v,
                np.where(np.isfinite(us), us, v),
            )
            nodes = np.concatenate([uv_nodes[uv_only], wl_ov, vis_nodes[vis_only]])
            values = np.concatenate([uv_scaled[uv_only], blend, vis_factor[vis_only]])

        finite = np.isfinite(values)
        nodes, values = nodes[finite], values[finite]
        order = np.argsort(nodes, kind="stable")
        nodes, values = nodes[order], values[order]
        ux, inv = np.unique(nodes, return_inverse=True)
        if ux.size != nodes.size:
            values = np.bincount(inv, weights=values, minlength=ux.size) / np.bincount(
                inv, minlength=ux.size
            )
            nodes = ux

        def _level(wl, spec, lo, hi):
            m = (wl > lo) & (wl < hi)
            return float(np.median(spec[m])) if m.any() else float("nan")

        info = dict(
            uv_range=tuple(float(x) for x in uv_range),
            vis_range=tuple(float(x) for x in vis_range),
            lowess_frac=float(lowess_frac),
            lowess_iters=int(lowess_iters),
            halogen_lowess_frac=halogen_lowess_frac,
            channel_edges_nm=list(channel_edges_nm) if channel_edges_nm else None,
            deuterium_median_counts_per_segment=[
                float(np.median(spec_d[channel_segments(wl_d, channel_edges_nm) == k]))
                for k in np.unique(channel_segments(wl_d, channel_edges_nm))
            ],
            halogen_median_counts_per_segment=[
                float(np.median(spec_h[channel_segments(wl_h, channel_edges_nm) == k]))
                for k in np.unique(channel_segments(wl_h, channel_edges_nm))
            ],
            overlap_fit=overlap_fit,
            overlap=overlap,
            n_overlap_pixels=int(ok.sum()),
            overlap_fit_rms=float(np.sqrt(np.mean((v[ok] - (intercept + slope * u[ok])) ** 2))),
            deuterium_median_counts_uv=_level(wl_d, spec_d, uv_range[0], ov_lo),
            deuterium_median_counts_overlap=_level(wl_d, spec_d, ov_lo, ov_hi),
            halogen_median_counts_overlap=_level(wl_h, spec_h, ov_lo, ov_hi),
            halogen_median_counts_vis=_level(wl_h, spec_h, ov_hi, vis_range[1]),
        )
        return cls(
            wavelength=nodes,
            factor=values,
            valid_range=(float(nodes.min()), float(nodes.max())),
            uv_scale=(intercept, slope),
            uv_nodes=uv_nodes,
            uv_factor=uv_factor,
            vis_nodes=vis_nodes,
            vis_factor=vis_factor,
            info=info,
        )

    # ---- use -----------------------------------------------------------------
    def __call__(self, wavelength: np.ndarray) -> np.ndarray:
        """Correction factor at ``wavelength`` [nm]; NaN outside ``valid_range``."""
        return interp_nan(np.asarray(wavelength, dtype=np.float64), self.wavelength, self.factor)

    def apply(
        self,
        spectra: np.ndarray,
        wavelength: np.ndarray,
        *,
        fill: float | str = np.nan,
    ) -> np.ndarray:
        """``spectra * factor(wavelength)`` along the last axis.

        Args:
            spectra:    (n_px,) or (n, n_px) intensities in pixel order
            wavelength: (n_px,) axis of the spectra [nm] (need not be monotonic)
            fill:       value of the factor outside the calibrated range: NaN
                        (default), a float (e.g. 0.0), or "edge" for the nearest
                        calibrated value
        Returns:
            corrected spectra, same shape, float64
        """
        wl = np.asarray(wavelength, dtype=np.float64)
        spectra = np.asarray(spectra, dtype=np.float64)
        if spectra.shape[-1] != wl.size:
            raise ValueError(f"spectra last axis {spectra.shape[-1]} != wavelength size {wl.size}")
        fac = self(wl)
        outside = ~np.isfinite(fac)
        if outside.any():
            if isinstance(fill, str):
                if fill != "edge":
                    raise ValueError("fill must be a number, NaN or 'edge'")
                fac = np.interp(wl, self.wavelength, self.factor)  # np.interp clamps to the edges
            else:
                fac = np.where(outside, float(fill), fac)
        return spectra * fac

    def sensitivity(self, wavelength: np.ndarray, normalise: bool = True) -> np.ndarray:
        """Relative instrument sensitivity ``1 / factor`` (max = 1 when ``normalise``)."""
        s = 1.0 / self(wavelength)
        if normalise:
            m = np.nanmax(s)
            if np.isfinite(m) and m > 0:
                s = s / m
        return s

    def sensitivity_curve(self, normalise: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """(wavelength, sensitivity) at the correction nodes."""
        return self.wavelength, self.sensitivity(self.wavelength, normalise)

    # ---- persistence ---------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Save nodes, factor and diagnostics to ``.npz`` (``load`` restores)."""
        import json

        np.savez(
            path,
            wavelength=self.wavelength,
            factor=self.factor,
            valid_range=np.asarray(self.valid_range),
            uv_scale=np.asarray(self.uv_scale),
            uv_nodes=self.uv_nodes,
            uv_factor=self.uv_factor,
            vis_nodes=self.vis_nodes,
            vis_factor=self.vis_factor,
            info=np.asarray(json.dumps(self.info, default=str)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "RelativeEfficiencyCorrection":
        import json

        with np.load(path, allow_pickle=False) as z:
            info = json.loads(str(z["info"])) if "info" in z else {}
            return cls(
                wavelength=z["wavelength"],
                factor=z["factor"],
                valid_range=tuple(float(v) for v in z["valid_range"]),
                uv_scale=tuple(float(v) for v in z["uv_scale"]),
                uv_nodes=z["uv_nodes"],
                uv_factor=z["uv_factor"],
                vis_nodes=z["vis_nodes"],
                vis_factor=z["vis_factor"],
                info=info,
            )

    def to_dataframe(self):
        """``pandas.DataFrame`` with columns wavelength, factor, sensitivity
        (the table the R script writes as ``FireFlyUvisCalibration.txt``)."""
        import pandas as pd

        return pd.DataFrame(
            {
                "wavelength": self.wavelength,
                "factor": self.factor,
                "sensitivity": self.sensitivity(self.wavelength),
            }
        )


def correct_spectra(
    spectra: np.ndarray,
    wavelength: np.ndarray,
    deuterium_h5: str | Path,
    halogen_h5: str | Path,
    *,
    fill: float | str = np.nan,
    **options: Any,
) -> np.ndarray:
    """One-call relative efficiency correction: build the correction from the two
    lamp files (options of :meth:`RelativeEfficiencyCorrection.from_arrays`) and
    apply it to ``spectra`` on ``wavelength``.  Rebuilding takes about a second;
    keep a :class:`RelativeEfficiencyCorrection` around for many calls."""
    rec = RelativeEfficiencyCorrection.from_lamp_files(deuterium_h5, halogen_h5, **options)
    return rec.apply(spectra, wavelength, fill=fill)


__all__ = [
    "RelativeEfficiencyCorrection",
    "correct_spectra",
    "load_lightigo_axis",
    "load_lightigo_spectrum",
    "lowess",
    "channel_segments",
    "fill_nonpositive",
    "interp_nan",
    "UV_LAMP_NM",
    "UV_LAMP_IRRADIANCE",
    "VIS_LAMP_NM",
    "VIS_LAMP_IRRADIANCE",
]
