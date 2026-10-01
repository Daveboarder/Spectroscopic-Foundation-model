"""Tests for data/canonical.py (canonical spectra shared by real maps and synthetic spectra)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import minimum_filter1d, uniform_filter1d

from data.canonical import (
    DEFAULT_CANONICAL,
    canonicalise,
    channel_bounds,
    patch_input,
    require_unit_norm,
    resolve_spec,
    unit_norm_rows,
)

FW = DEFAULT_CANONICAL["full_well_counts"]
SIGMA = np.array(DEFAULT_CANONICAL["noise_sigma_counts"])


def three_channel_axis() -> np.ndarray:
    """Avantes-like axis: 3 channels, each starting slightly below the previous end."""
    return np.concatenate(
        [
            np.arange(188.7, 260.01, 0.02),
            np.arange(259.99, 460.0, 0.026),
            np.arange(459.97, 689.1, 0.12),
        ]
    )


WL = three_channel_axis()
BOUNDS = channel_bounds(WL)


def line(centre: float, height: float, sigma: float = 0.08) -> np.ndarray:
    return height * np.exp(-0.5 * ((WL - centre) / sigma) ** 2)


def noise(rng, n=32, offset_counts=200.0) -> np.ndarray:
    x = np.empty((n, WL.size), np.float32)
    for k, (a, b) in enumerate(BOUNDS):
        x[:, a:b] = (offset_counts + rng.normal(0, SIGMA[k], (n, b - a))) / FW
    return x


def test_channel_bounds_follow_the_seams():
    assert len(BOUNDS) == 3
    assert BOUNDS[0][0] == 0 and BOUNDS[-1][1] == WL.size
    for (a, b), (c, d) in zip(BOUNDS[:-1], BOUNDS[1:]):
        assert b == c and WL[c] <= WL[b - 1]  # new channel where the axis stops increasing


@pytest.mark.parametrize("method", ["runmin", "median"])
def test_pedestal_removed_without_bleeding_across_seams(method):
    x = np.zeros((1, WL.size), np.float32)
    for k, (a, b) in enumerate(BOUNDS):
        x[0, a:b] = [0.002, 0.004, 0.007][k]  # different pedestal per channel
    x[0] += line(300.0, 0.3) + line(589.0, 0.5)
    c, _ = canonicalise(x, WL, {"baseline_method": method})
    far = np.ones(WL.size, bool)
    for centre in (300.0, 589.0):
        far &= np.abs(WL - centre) > 1.5
    assert np.abs(c[0, far]).max() == 0.0  # pedestals gone, including at the seams
    assert c[0, np.argmin(abs(WL - 300.0))] > 0 and c[0, np.argmin(abs(WL - 589.0))] > 0


def test_noise_floor_by_baseline_method():
    x = noise(np.random.default_rng(0))
    c_med, _ = canonicalise(x, WL, {"baseline_method": "median"})
    c_min, _ = canonicalise(x, WL, {"baseline_method": "runmin"})
    assert (c_med == 0).mean() >= 0.99  # baseline at the noise mean: 3 sigma removes it
    assert (c_min == 0).mean() >= 0.6  # running minimum sits below the mean: speckles remain
    # a line well above the floor survives either way
    y = x.copy()
    y[:, :] += line(400.0, 60 * SIGMA[1] / FW)
    c, _ = canonicalise(y, WL, {"baseline_method": "median"})
    assert (c[:, np.argmin(abs(WL - 400.0))] > 0).all()


def test_laser_blank_saturation_mask_and_l1_scale():
    x = (line(266.2, 0.2) + line(350.0, 0.2) + line(589.0, 5.0, sigma=0.4)).astype(np.float32)[None]
    x = np.minimum(x, 1.0)  # clipped plateau at full well
    c, sat = canonicalise(x, WL)
    laser = (WL > 265.4) & (WL < 267.0)
    assert (c[0, laser] == 0).all()
    assert sat[0].sum() > 3 and (x[0, sat[0]] >= 0.99).all()
    assert c[0, ~sat[0]].sum() / WL.size == pytest.approx(
        1.0, rel=1e-4
    )  # mean unsaturated pixel = 1


def test_matches_the_linear_screen_reference_on_unsaturated_spectra():
    """The e0 'canon' step of the linear transfer screen (counts in, floor in counts)."""
    rng = np.random.default_rng(1)
    counts = noise(rng, n=4) * FW + (line(300.0, 3000.0) + line(500.0, 8000.0))[None]
    counts = counts.astype(np.float32)
    ref = np.zeros_like(counts)
    for k, (a, b) in enumerate(BOUNDS):
        px = np.median(np.diff(WL[a:b]))
        win = max(5, int(round(2.0 / px)))
        base = uniform_filter1d(minimum_filter1d(counts[:, a:b], win, axis=1), win, axis=1)
        ref[:, a:b] = np.maximum(counts[:, a:b] - base - 3 * SIGMA[k], 0)
    ref[:, (WL > 265.4) & (WL < 267.0)] = 0
    ref = ref / ref.sum(1, keepdims=True) * ref.shape[1]
    got = patch_input(counts, WL, "counts", {"preprocess": "canonical"})[:, : WL.size]
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-5)


def test_patch_input_modes():
    rng = np.random.default_rng(2)
    u = unit_norm_rows(rng.random((3, WL.size)))
    assert np.array_equal(patch_input(u, WL, "unit_norm", {"preprocess": "none"}), u)  # idempotent
    fw = rng.random((3, WL.size)).astype(np.float32) * 0.5
    out = patch_input(fw, WL, "full_well", {"preprocess": "canonical"})
    assert out.shape == (3, 2 * WL.size)
    assert set(np.unique(out[:, WL.size :])) <= {0.0, 1.0}  # the mask half
    np.testing.assert_allclose(
        patch_input(fw * FW, WL, "counts", {"preprocess": "canonical"}), out, rtol=1e-5, atol=1e-6
    )
    with pytest.raises(ValueError, match="absolute scale"):
        patch_input(u, WL, "unit_norm", {"preprocess": "canonical"})


def test_resolve_spec_and_unit_guard():
    spec = resolve_spec(
        {"patch": {"preprocess": "canonical", "canonical": {"k_sigma": 4}}},
        {"generation": {"detector": {"full_well_counts": 60000}}},
    )
    assert spec["preprocess"] == "canonical"
    assert spec["canonical"]["k_sigma"] == 4 and spec["canonical"]["full_well_counts"] == 60000.0
    assert resolve_spec({})["preprocess"] == "none"
    with pytest.raises(ValueError):
        resolve_spec({"patch": {"preprocess": "bogus"}})

    class DS:
        units = "full_well"

    with pytest.raises(ValueError, match="unit-normalised"):
        require_unit_norm(DS(), "test")
    require_unit_norm(object(), "legacy datasets without the attribute are unit_norm")
