"""Cache keys of the shipped synthetic data configs must not change when the generator
gains opt-in options (keys recorded 2026-09-30 at commit 2281ae9; the full-matrix mineral
configs re-recorded 2026-10-01 after Li was added to muscovite in REE_minerals_oxides.xlsx,
the Avantes configs after the line-shape / self-absorption calibration of that day, and both
again after muscovite Li was lowered to 0.1 wt %).

The keys hash the loaded compositions, so editing the sample matrix changes them on purpose:
re-record the affected entries then. Builds each dataset with generation patched out, so
nothing is generated or read except the sample matrix, the line DB and the wavelength axis;
skipped when those are absent."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import data.libs_pipeline as lp
import data.two_zone_pipeline as tz

ROOT = Path(__file__).resolve().parent.parent
EXPECTED = {  # config: (synthetic cache key, dataset key incl. extra_spectra)
    "libs_data.yaml": ("179cf5839f60", "179cf5839f60"),
    "libs_data_cf_smoke.yaml": ("bfb3010aae3e", "bfb3010aae3e"),
    "libs_data_minerals.yaml": ("0898310046e4", "0898310046e4"),
    "libs_data_minerals_avantes.yaml": ("771cd30e6dc3", "6f2a8b325ca1"),
    "libs_data_minerals_avantes_smoke.yaml": ("7c7720a7aec2", "39f0c821a7f6"),
    "libs_data_minerals_smoke.yaml": ("806428455695", "806428455695"),
    "libs_data_smoke.yaml": ("7e74d4fdb7ad", "7e74d4fdb7ad"),
}


def _inputs_available(cfg: dict) -> bool:
    paths = cfg.get("paths") or {}
    files = [paths.get(k) for k in ("db", "sample_matrix", "wavelength_json")]
    files += [e.get("path") for e in cfg.get("extra_spectra") or []]
    return all(f and (ROOT / f if not Path(f).is_absolute() else Path(f)).is_file() for f in files)


def _keys(name: str, monkeypatch) -> tuple[str, str]:
    empty = lambda self: (pd.DataFrame(), np.empty((0, 0)))  # noqa: E731
    monkeypatch.setattr(lp.SyntheticLIBSDataset, "_build", empty)
    monkeypatch.setattr(tz.TwoZoneSyntheticDataset, "_build_synthetic", empty)
    monkeypatch.chdir(ROOT)
    cfg = yaml.safe_load(open(ROOT / "config" / name))
    if not _inputs_available(cfg):
        pytest.skip(f"inputs of {name} not available")
    cfg.setdefault("generation", {})["verbose"] = False
    ds = lp.build_dataset_from_config(cfg)
    return getattr(ds, "synthetic_cache_key", ds.cache_key), ds.cache_key


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_shipped_cache_keys_unchanged(name, monkeypatch):
    assert _keys(name, monkeypatch) == EXPECTED[name]


def test_full_well_variant_has_its_own_keys(monkeypatch):
    syn, key = _keys("libs_data_minerals_avantes_fw.yaml", monkeypatch)
    assert syn != EXPECTED["libs_data_minerals_avantes.yaml"][0]
    assert key != EXPECTED["libs_data_minerals_avantes.yaml"][1]


def test_mixtures_share_the_pure_cache_but_not_the_dataset_key(monkeypatch):
    syn, key = _keys("libs_data_minerals_avantes_fw.yaml", monkeypatch)
    syn_mix, key_mix = _keys("libs_data_minerals_avantes_fw_mix.yaml", monkeypatch)
    assert syn_mix == syn and key_mix != key
