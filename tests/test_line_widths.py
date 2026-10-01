"""Tests for the opt-in line-width options of data/two_zone_pipeline.py:
``generation.stark.hydrogen`` (Balmer Stark widths) and ``instrument.fwhm_ranges_nm``
(per-channel instrument FWHM). Both must leave spectra unchanged when absent.

The synthesis tests need the line DB external_data/Source/LIBS_data.db (skipped if absent).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import data.plasma_physics as pp
import data.two_zone_pipeline as tz

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "external_data" / "Source" / "LIBS_data.db"
STARK = {"hydrogen": "gigosos2003"}
needs_db = pytest.mark.skipif(not DB.is_file(), reason="line DB not available")


def fwhm(y: np.ndarray, x: np.ndarray, centre: float) -> float:
    """Width of the contiguous region above half maximum around the peak nearest ``centre``."""
    near = np.abs(x - centre) < 0.2
    i = int(np.flatnonzero(near)[np.argmax(y[near])])
    above = y >= y[i] / 2.0
    lo, hi = i, i
    while lo > 0 and above[lo - 1]:
        lo -= 1
    while hi < y.size - 1 and above[hi + 1]:
        hi += 1
    return float(x[hi] - x[lo])


def test_balmer_widths_follow_gigosos_2003():
    assert 2 * pp.balmer_stark_hwhm_nm(12.0875, 1e17) == pytest.approx(1.098)  # H alpha
    assert 2 * pp.balmer_stark_hwhm_nm(12.7485, 1e17) == pytest.approx(4.800)  # H beta
    ratio = pp.balmer_stark_hwhm_nm(12.0875, 3e17) / pp.balmer_stark_hwhm_nm(12.0875, 1e17)
    assert ratio == pytest.approx(3**0.67903)


def _zones(Ne: float = 3e17, l_inner: float = 1e-4):
    row = dict(plasma_model="one_zone", Te1=12500.0, Ne1=Ne, Te2=12500.0, Ne2=Ne, l_inner=l_inner,
               l_outer=0.0, N1=1e18, N2=1e18, gamma_stark1=0.005, gamma_stark2=0.005)  # fmt: skip
    return tz.zone_states_from_row(row)


@needs_db
def test_hydrogen_stark_broadens_h_alpha_and_keeps_its_area():
    grid = tz.make_fine_grid(np.linspace(630.0, 680.0, 501), 0.002)
    inner, outer = _zones()
    els, x = ["H", "O"], np.array([0.1, 0.9])
    off = tz.synthesise_fine_grid(els, x, grid, inner, outer, str(DB))
    on = tz.synthesise_fine_grid(els, x, grid, inner, outer, str(DB), stark=STARK)
    m = np.abs(grid - 656.28) < 25.0
    assert fwhm(off[m], grid[m], 656.28) < 0.1
    assert fwhm(on[m], grid[m], 656.28) == pytest.approx(
        2 * pp.balmer_stark_hwhm_nm(12.0875, 3e17), rel=0.03
    )
    # thin line: the area moves into the wings (+-20 HWHM window keeps ~97 % of a Lorentzian)
    assert np.trapezoid(on[m], grid[m]) / np.trapezoid(off[m], grid[m]) == pytest.approx(
        1.0, abs=0.04
    )


@needs_db
def test_hydrogen_stark_leaves_other_elements_bit_identical():
    grid = tz.make_fine_grid(np.linspace(380.0, 420.0, 401), 0.002)
    inner, outer = _zones()
    els, x = ["Al", "Ca", "Si", "O"], np.array([0.1, 0.02, 0.28, 0.6])
    a = tz.synthesise_fine_grid(els, x, grid, inner, outer, str(DB))
    b = tz.synthesise_fine_grid(els, x, grid, inner, outer, str(DB), stark=STARK)
    assert np.array_equal(a, b)


def test_instrument_ranges_set_the_width_per_channel():
    step = 0.002
    grid = np.arange(250.0, 650.0, step)
    rad = np.zeros_like(grid)
    for c in (300.0, 600.0):
        rad[np.argmin(np.abs(grid - c))] = 1.0
    inst = {"profile": "gaussian", "fwhm_nm": 0.15}
    plain = tz.broaden_instrument(rad, grid, step, inst)
    assert np.array_equal(plain, tz.apply_instrument(rad, tz.instrument_kernel(step, inst)))
    ranged = tz.broaden_instrument(rad, grid, step, {**inst, "fwhm_ranges_nm": [[460, 900, 0.26]]})
    w = {c: np.abs(grid - c) < 2.0 for c in (300.0, 600.0)}
    assert fwhm(ranged[w[300.0]], grid[w[300.0]], 300.0) == pytest.approx(0.15, abs=2 * step)
    assert fwhm(ranged[w[600.0]], grid[w[600.0]], 600.0) == pytest.approx(0.26, abs=2 * step)
    assert np.trapezoid(ranged[w[600.0]], grid[w[600.0]]) == pytest.approx(step, rel=1e-3)
