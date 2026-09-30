"""Tests for the detector model and the measured extra class of
data/two_zone_pipeline.py (``generation.detector``, ``extra_spectra``).

The lamp-file test needs external_data/Data/{Deuterium,Halogen}WithCorrection.h5
(gitignored) and is skipped when the files are absent.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from data.two_zone_pipeline import (
    apply_detector,
    detector_sensitivity,
    load_extra_spectra,
    response_jitter,
)

ROOT = Path(__file__).resolve().parent.parent
D2 = ROOT / "external_data" / "Data" / "DeuteriumWithCorrection.h5"
HAL = ROOT / "external_data" / "Data" / "HalogenWithCorrection.h5"
WL = np.linspace(190.0, 690.0, 5001)


def _line(centre: float, height: float = 1.0, sigma: float = 0.06) -> np.ndarray:
    return height * np.exp(-0.5 * ((WL - centre) / sigma) ** 2)


def test_response_multiplies_by_sensitivity_without_jitter():
    spec = _line(250.0) + _line(600.0)
    sens = np.where(WL < 400.0, 0.01, 1.0)
    det = {"response": {"jitter_dex": 0.0}}
    out = apply_detector(spec, WL, det, sens, np.random.default_rng(0))
    i250, i600 = np.argmin(abs(WL - 250.0)), np.argmin(abs(WL - 600.0))
    assert out.max() == pytest.approx(1.0)  # peak set to 1 without saturation
    assert out[i250] / out[i600] == pytest.approx(0.01, rel=1e-6)


def test_response_jitter_is_smooth_and_larger_where_untrusted():
    cfg = {
        "jitter_knot_nm": 25.0, "jitter_dex": 0.05, "jitter_dex_untrusted": 0.5,
        "untrusted_nm": [[190, 260]],
    }  # fmt: skip
    draws = np.log10([response_jitter(WL, cfg, np.random.default_rng(s)) for s in range(400)])
    sd = draws.std(axis=0)
    assert sd[WL < 250].mean() > 5 * sd[WL > 300].mean()
    assert np.abs(np.diff(draws, axis=1)).max() < 0.05  # piecewise linear over 25 nm knots


def test_saturation_clips_and_laser_line_is_added():
    spec = _line(500.0)
    det = {
        "saturation": {"log10_peak_over_saturation": [1.0, 1.0]},
        "laser_line": {"centre_nm": 266.18, "fwhm_nm": [0.2, 0.2], "amplitude": [0.3, 0.3]},
    }
    out = apply_detector(spec, WL, det, None, np.random.default_rng(0))
    assert out.max() == pytest.approx(1.0)
    assert (out >= 1.0).sum() > 1  # flat-topped (clipped) line, 10x over full well
    assert out[np.argmin(abs(WL - 266.18))] == pytest.approx(0.3, rel=0.05)


def test_unsaturated_peak_below_full_well():
    det = {"saturation": {"log10_peak_over_saturation": [-0.5, -0.5]}}
    out = apply_detector(_line(500.0), WL, det, None, np.random.default_rng(0))
    assert out.max() == pytest.approx(10**-0.5)


@pytest.mark.skipif(not (D2.is_file() and HAL.is_file()), reason="lamp HDF5 files not available")
def test_lamp_sensitivity_uv_far_below_vis():
    sens = detector_sensitivity(WL, {"deuterium_h5": str(D2), "halogen_h5": str(HAL)})
    assert sens.max() == pytest.approx(1.0)
    assert np.all(np.isfinite(sens)) and np.all(sens > 0)
    assert sens[WL < 260].mean() < 0.05 * sens[(WL > 550) & (WL < 650)].mean()


def test_load_extra_spectra(tmp_path):
    raw = np.stack([100.0 + _line(500.0, 400.0), 90.0 + _line(463.0, 50.0)]).astype(np.float32)
    path = tmp_path / "epoxy.h5"
    with h5py.File(path, "w") as f:
        f["wavelength"] = WL
        f["spectra"] = raw
    cols = ["sample_type_id", "sample_type_name", "unique_id", "C", "H", "O", "Si",
            "Te1", "Ne1", "N1", "l_inner", "plasma_model"]  # fmt: skip
    entry = {"path": str(path), "label": "epoxid", "composition": {"C": 3, "H": 0.5, "O": 1.5}}
    table, spectra = load_extra_spectra(entry, WL, cols)
    assert list(table.columns) == cols and len(table) == 2
    assert set(table["sample_type_name"]) == {"epoxid"} and table["unique_id"].is_unique
    assert table.loc[0, ["C", "H", "O", "Si"]].tolist() == pytest.approx([0.6, 0.1, 0.3, 0.0])
    assert (table[["Te1", "Ne1", "N1", "l_inner"]].to_numpy() == 0).all()  # no plasma labels
    assert spectra.min(axis=1) == pytest.approx([0, 0]) and spectra.max(axis=1) == pytest.approx(
        [1, 1]
    )
    with pytest.raises(ValueError, match="wavelength axis differs"):
        load_extra_spectra(entry, WL[:-1], cols)
