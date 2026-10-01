"""Tests for the boundary mixtures of data/two_zone_pipeline.py (``generation.mixtures``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import data.two_zone_pipeline as tz
from data.libs_pipeline import ZONE_COLUMNS


def pure_table() -> pd.DataFrame:
    """Two pure sample types with 3 shots each, contract-C1 columns."""
    rows = []
    for name, sid, comp in (("Albite", "ALBITE", {"Na": 0.09, "Al": 0.1, "Si": 0.32, "O": 0.49}),
                            ("Quartz", "QUARTZ", {"Si": 0.47, "O": 0.53})):  # fmt: skip
        for k in range(3):
            rec = {"sample_type_id": sid, "sample_type_name": name, "unique_id": f"{sid}_{k}"}
            rec.update({e: comp.get(e, 0.0) for e in ("Na", "Al", "Si", "O")})
            rec.update(Te=12500.0 + k, Ne=3e17)
            rec.update({c: 1.0 + k for c in ZONE_COLUMNS})
            rec["plasma_model"] = "two_zone"
            rows.append(rec)
    return pd.DataFrame(rows)


def test_pairs_from_minerals_or_explicit():
    assert tz.mixture_pairs({"minerals": ["A", "B", "C"]}) == [("A", "B"), ("A", "C"), ("B", "C")]
    assert tz.mixture_pairs({"pairs": [["X", "Y"]], "minerals": ["A", "B"]}) == [("X", "Y")]


def test_mixture_rows_mix_compositions_and_carry_no_plasma_labels():
    pure = pure_table()
    cfg = {"minerals": ["Albite", "Quartz"], "n_samples_per_pair": 50, "fraction": [0.2, 0.8]}
    table, tasks = tz.build_mixture_table(pure, cfg, seed=1)
    assert list(table.columns) == list(pure.columns) and len(table) == len(tasks) == 50
    assert set(table["sample_type_name"]) == {"Albite + Quartz"}
    assert set(table["sample_type_id"]) == {"MIX_ALBITE_QUARTZ"}
    assert (table["plasma_model"] == "mixture").all()
    assert (table[[c for c in ZONE_COLUMNS if c != "plasma_model"]] == 0).all().all()
    els = ["Na", "Al", "Si", "O"]
    assert np.allclose(table[els].sum(axis=1), 1.0)
    w = np.array([float(u.rsplit("_w", 1)[1]) for u in table["unique_id"]])
    assert w.min() >= 0.2 and w.max() <= 0.8
    # Na comes only from albite: its mass fraction is w * 0.09
    assert np.allclose(table["Na"], w * 0.09, atol=6e-4)  # w is rounded to 3 decimals in unique_id
    (ea, ca, ra), (eb, cb, rb), w0 = tasks[0]
    assert ea == els and ca[0] == pytest.approx(0.09) and cb[0] == 0.0 and 0.2 <= w0 <= 0.8
    again, _ = tz.build_mixture_table(pure, cfg, seed=1)
    assert again.equals(table)  # deterministic
    with pytest.raises(ValueError, match="not a generated sample type"):
        tz.build_mixture_table(pure, {"minerals": ["Albite", "Beryl"]}, seed=1)


def test_mixture_shot_is_the_weighted_radiance_sum_with_one_detector_pass(monkeypatch):
    wl = np.linspace(300.0, 310.0, 101)
    rad = {
        "A": np.where(np.arange(101) == 30, 4.0, 0.0),
        "B": np.where(np.arange(101) == 70, 2.0, 0.0),
    }
    monkeypatch.setattr(
        tz, "synthesise_radiance", lambda el, mf, w, row, db, gc=None: rad[row["who"]].copy()
    )
    comps = ((["Si"], np.ones(1), {"who": "A"}), (["Si"], np.ones(1), {"who": "B"}))
    tz._init_zone_worker(wl, "unused.db", {"seed": 0})
    _, spec = tz._generate_mixture_one((0, comps, 0.25))
    assert np.allclose(spec, tz.unit_norm(0.25 * rad["A"] + 0.75 * rad["B"]))
    # full-well output with saturation: the detector scales and clips the SUM once
    det = {"output_units": "full_well", "saturation": {"log10_peak_over_saturation": [0.5, 0.5]}}
    tz._init_zone_worker(wl, "unused.db", {"seed": 0, "detector": det})
    _, spec = tz._generate_mixture_one((0, comps, 0.5))
    assert spec.max() == pytest.approx(1.0) and spec[30] == spec[70] == pytest.approx(1.0)
