"""Tests for data/efficiency_correction.py (relative efficiency correction).

The lamp-file tests need external_data/Data/{Deuterium,Halogen}WithCorrection.h5
(gitignored) and are skipped when the files are absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data.efficiency_correction import (
    UV_LAMP_IRRADIANCE,
    UV_LAMP_NM,
    VIS_LAMP_IRRADIANCE,
    VIS_LAMP_NM,
    RelativeEfficiencyCorrection,
    channel_segments,
    correct_spectra,
    fill_nonpositive,
    interp_nan,
    lowess,
)

ROOT = Path(__file__).resolve().parent.parent
D2 = ROOT / "external_data" / "Data" / "DeuteriumWithCorrection.h5"
HAL = ROOT / "external_data" / "Data" / "HalogenWithCorrection.h5"
needs_lamps = pytest.mark.skipif(
    not (D2.is_file() and HAL.is_file()), reason="lamp HDF5 files not available"
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def test_interp_nan_ties_and_range():
    y = interp_nan([0.5, 1.0, 2.5, 3.0], [0, 1, 1, 2, 3], [0, 2, 4, 6, 9])
    assert y[0] == pytest.approx(1.5)  # between 0 and the tie mean 3
    assert y[1] == pytest.approx(3.0)  # ties averaged
    assert y[2] == pytest.approx(7.5)
    assert y[3] == pytest.approx(9.0)
    assert np.isnan(interp_nan([-1.0, 4.0], [0, 1, 2, 3], [0, 1, 2, 3])).all()


def test_fill_nonpositive_interpolates_neighbours():
    y = fill_nonpositive([5.0, 0.0, 0.0, 8.0, 0.0])
    assert y.tolist() == pytest.approx([5.0, 6.0, 7.0, 8.0, 8.0])
    with pytest.raises(ValueError):
        fill_nonpositive([0.0, 0.0])


def test_lowess_recovers_smooth_signal_and_rejects_outliers():
    rng = np.random.default_rng(0)
    n = 2000
    x = np.arange(n)
    truth = 100.0 + 0.05 * x + 20.0 * np.sin(x / 300.0)
    y = truth + rng.normal(0.0, 3.0, n)
    y[::97] += 200.0  # spikes
    sm = lowess(y, frac=0.05, iters=3)
    inner = slice(60, n - 60)
    assert np.sqrt(np.mean((sm[inner] - truth[inner]) ** 2)) < 1.5
    assert np.max(np.abs(sm[inner] - truth[inner])) < 6.0


# ---------------------------------------------------------------------------
# synthetic end-to-end: known sensitivity must be recovered
# ---------------------------------------------------------------------------
def _synthetic_lamp_measurements(sens):
    """Two lamp 'measurements' on a 3-segment non-monotonic axis with a known sensitivity."""
    wl = np.concatenate(
        [
            np.linspace(189, 370.1, 3000),
            np.linspace(370.0, 460.05, 1500),
            np.linspace(459.98, 859, 4000),
        ]
    )
    lamp_uv = np.interp(wl, UV_LAMP_NM, UV_LAMP_IRRADIANCE)
    lamp_vis = np.interp(wl, VIS_LAMP_NM, VIS_LAMP_IRRADIANCE)
    s = sens(wl)
    d2 = 3000.0 * lamp_uv * s  # bright deuterium in the UV, dies out in the VIS
    d2[wl > 420] *= np.exp(-(wl[wl > 420] - 420) / 40.0)
    hal = 800.0 * lamp_vis * s  # halogen weak in the UV
    hal[wl < 330] *= np.exp(-(330 - wl[wl < 330]) / 30.0)
    d2[[0, 1500, 3000, 3001, 4500, 4501]] = 0.0  # dead pixels / seams
    hal[[0, 3000, 3001, 4500]] = 0.0
    return wl, d2, hal


def _sens(wl):
    return 0.05 + np.exp(-0.5 * ((wl - 520.0) / 150.0) ** 2)


def _sens_step(wl):
    """Sensitivity with x5 gain steps at 260 and 690 nm (FireFly-like channels)."""
    wl = np.asarray(wl, dtype=np.float64)
    return _sens(wl) * np.where(wl < 260.0, 0.2, 1.0) * np.where(wl >= 690.0, 5.0, 1.0)


def test_channel_segments():
    wl = np.array([200.0, 250.0, 259.9, 260.0, 300.0, 371.0, 370.5, 380.0])
    seg = channel_segments(wl, (260.0,), min_segment_px=1)
    assert seg.tolist() == [0, 0, 0, 1, 1, 1, 2, 2]
    assert channel_segments(wl, None, min_segment_px=1).tolist() == [0, 0, 0, 0, 0, 0, 1, 1]
    # short segments are merged into the preceding one by default
    assert channel_segments(wl, (260.0,)).tolist() == [0] * 8


def test_gain_steps_preserved_by_per_channel_smoothing():
    wl, d2, hal = _synthetic_lamp_measurements(_sens_step)
    rec = RelativeEfficiencyCorrection.from_arrays(wl, d2, wl, hal)
    q = np.concatenate([np.linspace(200, 258, 60), np.linspace(262, 850, 400)])
    est = rec.sensitivity(q, normalise=False)
    true = _sens_step(q)
    ratio = est / true
    ratio = ratio / np.nanmedian(ratio)
    assert np.nanmax(np.abs(ratio - 1.0)) < 0.05
    # smoothing across the step (no channel edges) is visibly wrong next to it
    rec_flat = RelativeEfficiencyCorrection.from_arrays(wl, d2, wl, hal, channel_edges_nm=None)
    near = np.array([258.5, 261.5])
    r2 = rec_flat.sensitivity(near, normalise=False) / _sens_step(near)
    assert np.max(np.abs(r2 / np.nanmedian(ratio) - 1.0)) > 0.05


@pytest.mark.parametrize("overlap", ["crossfade", "interleave"])
def test_recovers_known_sensitivity(overlap):
    wl, d2, hal = _synthetic_lamp_measurements(_sens)
    rec = RelativeEfficiencyCorrection.from_arrays(wl, d2, wl, hal, overlap=overlap)
    q = np.linspace(200, 850, 500)
    est = rec.sensitivity(q, normalise=True)
    true = _sens(q) / np.nanmax(_sens(rec.wavelength))
    ratio = est / true
    # shape recovered to a few percent everywhere; a constant factor is allowed
    ratio = ratio / np.nanmedian(ratio)
    assert np.nanmax(np.abs(ratio - 1.0)) < 0.05
    assert rec.valid_range[0] < 200 and rec.valid_range[1] > 850
    assert np.isnan(rec(100.0)) and np.isnan(rec(950.0))


def test_apply_shapes_and_fill():
    wl, d2, hal = _synthetic_lamp_measurements(_sens)
    rec = RelativeEfficiencyCorrection.from_arrays(wl, d2, wl, hal)
    axis = np.linspace(150, 900, 400)  # extends beyond the calibrated range
    spectra = np.ones((3, axis.size))
    out = rec.apply(spectra, axis)
    assert out.shape == spectra.shape
    assert np.isnan(out[:, 0]).all() and np.isnan(out[:, -1]).all()
    out0 = rec.apply(spectra[0], axis, fill=0.0)
    assert out0.shape == (axis.size,) and out0[0] == 0.0 and np.isfinite(out0).all()
    edge = rec.apply(spectra, axis, fill="edge")
    assert np.isfinite(edge).all()
    assert edge[0, 0] == pytest.approx(rec.factor[0])
    with pytest.raises(ValueError):
        rec.apply(np.ones(10), axis)


def test_save_load_roundtrip(tmp_path):
    wl, d2, hal = _synthetic_lamp_measurements(_sens)
    rec = RelativeEfficiencyCorrection.from_arrays(wl, d2, wl, hal)
    p = tmp_path / "rec.npz"
    rec.save(p)
    back = RelativeEfficiencyCorrection.load(p)
    q = np.linspace(200, 850, 50)
    assert np.allclose(back(q), rec(q))
    assert back.valid_range == rec.valid_range
    assert back.info["overlap"] == "crossfade"
    df = back.to_dataframe()
    assert list(df.columns) == ["wavelength", "factor", "sensitivity"]


# ---------------------------------------------------------------------------
# the real FireFly lamp files
# ---------------------------------------------------------------------------
@needs_lamps
def test_lamp_files_build_and_self_consistency():
    rec = RelativeEfficiencyCorrection.from_lamp_files(D2, HAL)
    assert 188 < rec.valid_range[0] < 190 and 855 < rec.valid_range[1] < 860
    assert np.isfinite(rec.factor).all() and (rec.factor > 0).all()
    # correcting the halogen measurement itself must give back the certified halogen shape
    from data.efficiency_correction import load_lightigo_spectrum

    wl, hal, _ = load_lightigo_spectrum(HAL)
    corr = rec.apply(hal, wl)
    m = (wl > 420) & (wl < 850)
    ratio = corr[m] / np.interp(wl[m], VIS_LAMP_NM, VIS_LAMP_IRRADIANCE)
    ratio = ratio / np.median(ratio)
    assert np.percentile(np.abs(ratio - 1.0), 95) < 0.05
    # and the convenience function agrees with the class
    corr2 = correct_spectra(hal, wl, D2, HAL)
    assert np.allclose(np.nan_to_num(corr2), np.nan_to_num(corr))
