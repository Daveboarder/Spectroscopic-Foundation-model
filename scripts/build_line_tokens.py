"""
Standalone CLI that builds the line-token HDF5 cache from scratch.

This runs the full preprocessing pipeline once and writes:
    external_data/cache/line_dict_<hash>.h5
    external_data/cache/line_features_<hash>.h5
    external_data/cache/line_tokens_<hash>.h5

The resulting line_tokens_*.h5 is the only artefact required at training time
for ``embedding_type: line_token_linear``. It can be reused across runs and
across different model configs (provided the line_embedding_config has not
changed — the file path encodes a content hash).

Usage:
    uv run python scripts/build_line_tokens.py \\
        --libs_data_config config/libs_data.yaml \\
        --line_embedding_config config/line_embedding.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.canonical import require_unit_norm
from data.libs_pipeline import build_dataset_from_config
from data.line_embedding_pipeline import prepare_line_tokens_assets


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Tokenize per-spectrum line features into a single reusable HDF5 "
            "cache (combining the theoretical dictionary with the Voigt fits). "
            "Safe to run repeatedly — already-built caches are reused."
        )
    )
    parser.add_argument("--libs_data_config", type=str, required=True)
    parser.add_argument("--line_embedding_config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    libs_cfg = yaml.safe_load(open(args.libs_data_config))
    libs_cfg.setdefault("generation", {}).setdefault("seed", args.seed)

    ds = build_dataset_from_config(libs_cfg)
    require_unit_norm(ds, "build_line_tokens")
    if len(ds) == 0:
        raise SystemExit("LIBS pipeline produced no spectra — check configs.")

    meta = prepare_line_tokens_assets(
        ds.spectra.astype(np.float32, copy=False),   # already float32 from the cache
        ds.wavelength,
        args.line_embedding_config,
        spectra_cache_key=ds.cache_key,
        verbose=True,
    )

    print()
    print("=" * 60)
    print("Line-token assets ready")
    print("=" * 60)
    print(f"  Dictionary: {meta['line_dict_path']}")
    print(f"  Voigt fits: {meta['line_features_path']}")
    print(f"  Tokens:     {meta['line_tokens_path']}")
    print(f"  Shape:      [{ds.spectra.shape[0]}, {meta['n_lines']}, {meta['n_features']}]")
    print(f"  Features:   {meta['feature_names']}")
    print()
    print("Use this cache for training with:")
    print(
        "  uv run python train_pretrain.py \\\n"
        "      --config config/config_libs_token_linear_4090.yaml \\\n"
        f"      --libs_data_config {args.libs_data_config} \\\n"
        f"      --line_embedding_config {args.line_embedding_config}"
    )


if __name__ == "__main__":
    main()
