"""
Pre-training script for LIBS Foundation Model.

Creates a timestamped run directory, generates synthetic data,
trains the transformer with MIP, and saves checkpoints.

Usage:
    uv run python train_pretrain.py --config config/config_a100.yaml --experiment_name big_pretrain
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
    Callback,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from data.synthetic_generator import SyntheticLIBSGenerator
from data.libs_pipeline import build_dataset_from_config
from data.line_embedding_pipeline import (
    prepare_line_token_assets,
    prepare_line_tokens_assets,
)
from data.discretization import build_discretizer_from_config
from models.libs_transformer import LIBSTransformer
from models.spectral_patch_embedding import spectral_meta_from_config, spectral_patch_run_info
from training.pretrain import LIBSPretrainModule, PretrainDataModule
from utils.run_manager import RunManager


def pretrain_loss_type(config: dict) -> str:
    """``mse`` (default) or ``classification``."""
    return str(config.get('pretrain', {}).get('loss', 'mse')).lower()


def build_pretrain_discretizers(config: dict):
    """Build intensity (+ FWHM) discretizers when using classification MIP."""
    if pretrain_loss_type(config) != 'classification':
        return None, None
    disc_root = config.get('discretization', {})
    intensity_cfg = disc_root.get('intensity') or {
        'num_bins': config['model'].get('num_intensity_bins', 256),
        'min_val': 1.0e-4,
        'max_val': 1.0,
        'strategy': 'log',
    }
    fwhm_cfg = disc_root.get('fwhm') or {
        'num_bins': config['model'].get('num_fwhm_bins', 100),
        'min_val': 0.005,
        'max_val': 0.3,
        'strategy': 'uniform',
    }
    return (
        build_discretizer_from_config(intensity_cfg),
        build_discretizer_from_config(fwhm_cfg),
    )


class SaveRawModelCallback(Callback):
    """Save raw model weights every N epochs for easy mid-training evaluation."""
    def __init__(self, save_path: str, save_every_n_epochs: int = 1):
        self.save_path = save_path
        self.save_every_n_epochs = save_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.save_every_n_epochs == 0:
            torch.save(pl_module.model.state_dict(), self.save_path)
            info_path = str(self.save_path).replace('.pt', '_info.txt')
            with open(info_path, 'w') as f:
                f.write(f"epoch: {trainer.current_epoch + 1}\n")
                f.write(f"global_step: {trainer.global_step}\n")
                val_loss = trainer.callback_metrics.get('val/loss')
                if val_loss is not None:
                    f.write(f"val_loss: {val_loss.item():.6f}\n")


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _generate_legacy_synthetic(config: dict, seed: int):
    """Original 5-class Gaussian-peak synthetic data (kept for sanity comparison)."""
    print("Generating legacy synthetic data...")
    generator = SyntheticLIBSGenerator(
        n_bins=config['data']['n_bins'],
        noise_sigma=config['data']['synthetic']['noise_sigma'],
        peak_width_range=tuple(config['data']['synthetic']['peak_width_range']),
        intensity_variation=config['data']['synthetic']['intensity_variation'],
        seed=seed,
    )
    train_spectra, _, _ = generator.generate_dataset(
        n_samples=config['data']['synthetic']['train_samples'], return_labels=True,
    )
    generator_val = SyntheticLIBSGenerator(
        n_bins=config['data']['n_bins'],
        noise_sigma=config['data']['synthetic']['noise_sigma'],
        peak_width_range=tuple(config['data']['synthetic']['peak_width_range']),
        intensity_variation=config['data']['synthetic']['intensity_variation'],
        seed=seed + 1000,
    )
    val_spectra, _, _ = generator_val.generate_dataset(
        n_samples=config['data']['synthetic']['val_samples'], return_labels=True,
    )
    return train_spectra, val_spectra


def _generate_libs_pipeline(
    config: dict, libs_config_path: str, seed: int, return_wavelength: bool = False,
):
    """Realistic physics-based synthesis (Daveboarder/Element_Identification heritage).

    The full dataset is materialised once (cached to HDF5), then split into
    train/val by `data.synthetic.val_fraction` (default 0.1). n_bins is
    overridden in-place to match the actual wavelength array length. With
    ``return_wavelength`` the spectrometer axis [nm] is returned as a third item
    (spectral_patch embedding)."""
    print(f"Generating LIBS-pipeline data from {libs_config_path}...")
    libs_cfg = yaml.safe_load(open(libs_config_path))
    libs_cfg.setdefault('generation', {})['seed'] = seed

    ds = build_dataset_from_config(libs_cfg)
    spectra = ds.spectra.astype(np.float32)
    if spectra.size == 0:
        raise RuntimeError("LIBS pipeline produced no spectra — check sample types / DB.")

    # Override model n_bins to match the real wavelength axis (e.g. 69712)
    actual_n_bins = spectra.shape[1]
    if config['data']['n_bins'] != actual_n_bins:
        print(f"Overriding n_bins: {config['data']['n_bins']} -> {actual_n_bins}")
        config['data']['n_bins'] = actual_n_bins
        config['model']['max_seq_len'] = actual_n_bins + 1

    val_frac = config['data']['synthetic'].get('val_fraction', 0.1)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(spectra))
    n_val = max(1, int(len(spectra) * val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    if return_wavelength:
        return spectra[train_idx], spectra[val_idx], np.asarray(ds.wavelength, dtype=np.float64)
    return spectra[train_idx], spectra[val_idx]


def generate_data(config: dict, seed: int = 42, libs_config_path: str | None = None):
    if libs_config_path:
        return _generate_libs_pipeline(config, libs_config_path, seed)
    train_spectra, val_spectra = _generate_legacy_synthetic(config, seed)
    print(f"Generated {len(train_spectra)} training samples")
    print(f"Generated {len(val_spectra)} validation samples")
    return train_spectra, val_spectra


def create_model(
    config: dict,
    line_dict_meta: dict | None = None,
    line_token_meta: dict | None = None,
    spectral_meta: dict | None = None,
) -> LIBSTransformer:
    emb_type = config['model'].get('embedding_type', 'intensity')
    kwargs = dict(
        n_bins=config['data']['n_bins'],
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        n_layers=config['model']['n_layers'],
        d_ff=config['model']['d_ff'],
        dropout=config['model']['dropout'],
        n_classes=config['data']['n_classes'],
        embedding_type=emb_type,
    )
    if emb_type == 'line_token' and line_dict_meta is not None:
        kwargs['n_lines'] = line_dict_meta['n_lines']
        kwargs['line_dict_meta'] = line_dict_meta
        kwargs['n_elements_vocab'] = line_dict_meta.get('n_elements', 53)
    if emb_type == 'line_token_linear' and line_token_meta is not None:
        kwargs['n_lines'] = line_token_meta['n_lines']
        kwargs['line_token_meta'] = line_token_meta
        kwargs['n_mip_target_channels'] = int(
            config['model'].get('n_mip_target_channels', 2)
        )
    if emb_type == 'spectral_patch':
        kwargs['spectral_meta'] = spectral_meta
    kwargs['mip_loss_type'] = pretrain_loss_type(config)
    kwargs['num_intensity_bins'] = int(config['model'].get('num_intensity_bins', 256))
    kwargs['num_fwhm_bins'] = int(config['model'].get('num_fwhm_bins', 100))
    model = LIBSTransformer(**kwargs)
    loss_label = kwargs['mip_loss_type']
    print(f"Created model ({emb_type}, MIP={loss_label}) with {model.num_parameters:,} parameters")
    return model


def _prepare_line_token_libs(
    config: dict, libs_config_path: str, line_embedding_config: str, seed: int,
):
    """Build caches and train/val index split for line-token + LIBS pipeline."""
    libs_cfg = yaml.safe_load(open(libs_config_path))
    libs_cfg.setdefault('generation', {})['seed'] = seed
    ds = build_dataset_from_config(libs_cfg)
    if len(ds) == 0:
        raise RuntimeError("LIBS pipeline produced no spectra.")
    meta = prepare_line_token_assets(
        ds.spectra.astype(np.float32),
        ds.wavelength,
        line_embedding_config,
        spectra_cache_key=ds.cache_key,
        verbose=True,
    )
    n_lines = meta['n_lines']
    print(f"Line dictionary: {n_lines} tokens")
    config['data']['n_bins'] = n_lines
    config['model']['max_seq_len'] = n_lines + 1

    val_frac = config['data']['synthetic'].get('val_fraction', 0.1)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ds))
    n_val = max(1, int(len(ds) * val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return meta, train_idx, val_idx


def _prepare_line_tokens_linear_libs(
    config: dict, libs_config_path: str, line_embedding_config: str, seed: int,
):
    """Pre-baked tokens path for the line_token_linear model."""
    libs_cfg = yaml.safe_load(open(libs_config_path))
    libs_cfg.setdefault('generation', {})['seed'] = seed
    ds = build_dataset_from_config(libs_cfg)
    if len(ds) == 0:
        raise RuntimeError("LIBS pipeline produced no spectra.")
    meta = prepare_line_tokens_assets(
        ds.spectra.astype(np.float32),
        ds.wavelength,
        line_embedding_config,
        spectra_cache_key=ds.cache_key,
        verbose=True,
    )
    print(
        f"Line tokens cache: {meta['n_lines']} lines × {meta['n_features']} features → "
        f"{meta['line_tokens_path']}"
    )
    config['data']['n_bins'] = meta['n_lines']
    config['model']['max_seq_len'] = meta['n_lines'] + 1

    val_frac = config['data']['synthetic'].get('val_fraction', 0.1)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ds))
    n_val = max(1, int(len(ds) * val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return meta, train_idx, val_idx


def main(args):
    config = load_config(args.config)

    run_mgr = RunManager(
        run_type="pretrain",
        experiment_name=args.experiment_name,
        base_dir=args.runs_dir,
        config_path=args.config,
    )

    pl.seed_everything(args.seed)

    line_dict_meta = None
    line_token_meta = None
    train_indices = val_indices = None
    line_features_path = None
    line_tokens_path = None
    train_spectra = val_spectra = None
    spectral_meta = None

    requested_emb = config['model'].get('embedding_type', 'intensity')
    # Default behaviour when --line_embedding_config is provided but the
    # model config does not pin a flavour: use the new linear/tokenized path.
    if args.line_embedding_config and requested_emb == 'intensity':
        requested_emb = 'line_token_linear'
        config['model']['embedding_type'] = requested_emb
    use_line_token = requested_emb in ('line_token', 'line_token_linear')

    if use_line_token:
        if not args.line_embedding_config:
            raise ValueError(f"embedding_type={requested_emb!r} requires --line_embedding_config")
        if not args.libs_data_config:
            raise ValueError(f"embedding_type={requested_emb!r} requires --libs_data_config")
    if requested_emb == 'spectral_patch':
        if args.line_embedding_config:
            raise ValueError("embedding_type='spectral_patch' reads raw spectra; "
                             "do not pass --line_embedding_config")
        if not args.libs_data_config:
            raise ValueError("embedding_type='spectral_patch' requires --libs_data_config")

    if requested_emb == 'line_token':
        line_dict_meta, train_indices, val_indices = _prepare_line_token_libs(
            config, args.libs_data_config, args.line_embedding_config, args.seed,
        )
        line_features_path = line_dict_meta['line_features_path']
        print(f"Train: {len(train_indices)} | Val: {len(val_indices)} | n_lines: {config['data']['n_bins']}")
    elif requested_emb == 'line_token_linear':
        line_token_meta, train_indices, val_indices = _prepare_line_tokens_linear_libs(
            config, args.libs_data_config, args.line_embedding_config, args.seed,
        )
        line_tokens_path = line_token_meta['line_tokens_path']
        print(f"Train: {len(train_indices)} | Val: {len(val_indices)} | n_lines: {config['data']['n_bins']}")
    elif requested_emb == 'spectral_patch':
        train_spectra, val_spectra, wavelength = _generate_libs_pipeline(
            config, args.libs_data_config, args.seed, return_wavelength=True,
        )
        spectral_meta = spectral_meta_from_config(config['model'], wavelength)
        print(f"Train: {len(train_spectra)} samples | Val: {len(val_spectra)} samples | "
              f"n_bins: {train_spectra.shape[1]} (spectral_patch)")
    else:
        train_spectra, val_spectra = generate_data(
            config, seed=args.seed, libs_config_path=args.libs_data_config,
        )
        print(f"Train: {len(train_spectra)} samples | Val: {len(val_spectra)} samples | "
              f"n_bins: {train_spectra.shape[1]}")

    if args.save_data and train_spectra is not None:
        data_dir = run_mgr.run_dir / "data"
        data_dir.mkdir(exist_ok=True)
        np.save(data_dir / 'train_spectra.npy', train_spectra)
        np.save(data_dir / 'val_spectra.npy', val_spectra)
        print(f"Saved data to {data_dir}")

    model = create_model(
        config, line_dict_meta=line_dict_meta, line_token_meta=line_token_meta,
        spectral_meta=spectral_meta,
    )

    loss_type = pretrain_loss_type(config)
    intensity_disc, fwhm_disc = build_pretrain_discretizers(config)
    if loss_type == 'classification':
        print(f"Pretrain loss: classification (intensity bins={intensity_disc.num_bins}, "
              f"fwhm bins={fwhm_disc.num_bins})")

    pretrain_module = LIBSPretrainModule(
        model=model,
        learning_rate=config['pretrain']['learning_rate'],
        weight_decay=config['pretrain']['weight_decay'],
        warmup_epochs=config['pretrain']['warmup_epochs'],
        max_epochs=config['pretrain']['epochs'],
        min_lr=config['pretrain']['min_lr'],
        loss_type=loss_type,
        intensity_discretizer=intensity_disc,
        fwhm_discretizer=fwhm_disc,
        mask_ratio=config['pretrain']['mask_ratio'],
        mask_span_tokens=int(config['pretrain'].get('mask_span_tokens', 1)),
    )

    # Masking config
    contiguous_masking = config['pretrain'].get('contiguous_masking', False) or args.contiguous_masking
    block_sizes_config = config['pretrain'].get('block_sizes', None)
    block_sizes = block_sizes_config if block_sizes_config else [config['pretrain'].get('contiguous_mask_size', 50)]
    peak_bias_enabled = config['pretrain'].get('peak_bias_enabled', False)
    peak_bias_ratio = config['pretrain'].get('peak_bias_ratio', 0.5)
    peak_threshold = config['pretrain'].get('peak_threshold', 0.2)

    print(f"\nMasking Configuration:")
    print(f"  Contiguous: {contiguous_masking}")
    print(f"  Block sizes: {block_sizes}")
    print(f"  Mask ratio: {config['pretrain']['mask_ratio']}")
    print(f"  Peak-biased: {peak_bias_enabled} (ratio: {peak_bias_ratio}, threshold: {peak_threshold})")
    if spectral_meta is not None:
        print(f"  spectral_patch: {model.embedding.n_tokens} windows, span "
              f"{config['pretrain'].get('mask_span_tokens', 1)} windows (masked on the GPU)")

    data_module = PretrainDataModule(
        train_spectra=train_spectra,
        val_spectra=val_spectra,
        batch_size=config['pretrain']['batch_size'],
        mask_ratio=config['pretrain']['mask_ratio'],
        contiguous_masking=contiguous_masking,
        block_sizes=block_sizes,
        peak_bias_enabled=peak_bias_enabled,
        peak_bias_ratio=peak_bias_ratio,
        peak_threshold=peak_threshold,
        num_workers=args.num_workers,
        line_features_path=line_features_path,
        line_tokens_path=line_tokens_path,
        train_indices=train_indices,
        val_indices=val_indices,
        spectral_patch=spectral_meta is not None,
    )

    # Logger
    logger_type = config['logging'].get('logger', 'tensorboard')

    if logger_type == 'wandb':
        wandb_config = config['logging'].get('wandb', {})
        logger = WandbLogger(
            project=wandb_config.get('project', 'libs-foundation-model'),
            entity=wandb_config.get('entity'),
            name=run_mgr.run_name,
            tags=wandb_config.get('tags', []) + ['pretrain'],
            save_dir=str(run_mgr.log_dir),
            config={
                'model': config['model'],
                'pretrain': config['pretrain'],
                'data': config['data'],
                'run_dir': str(run_mgr.run_dir),
            },
        )
    else:
        logger = TensorBoardLogger(
            save_dir=str(run_mgr.log_dir),
            name='',
            version='',
        )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(run_mgr.checkpoint_dir),
        filename='best',
        save_top_k=1,
        monitor='val/loss',
        mode='min',
        save_last=True,
        auto_insert_metric_name=False,
    )

    raw_model_callback = SaveRawModelCallback(
        save_path=str(run_mgr.checkpoint_dir / 'model_latest.pt'),
        save_every_n_epochs=1,
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    callbacks = [checkpoint_callback, lr_monitor, raw_model_callback]
    if args.early_stopping:
        callbacks.append(EarlyStopping(
            monitor='val/loss', patience=10, mode='min', verbose=True,
        ))

    # Trainer
    trainer_kwargs = dict(
        accelerator=config['device']['accelerator'],
        devices=1,
        precision=config['device']['precision'],
        max_epochs=config['pretrain']['epochs'],
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=config['logging']['log_every_n_steps'],
        val_check_interval=config['logging']['val_check_interval'],
        gradient_clip_val=1.0,
        deterministic=True,
    )
    # Gradient accumulation (for large effective batch on limited VRAM)
    accumulate = config['pretrain'].get('accumulate_grad_batches', None)
    if accumulate and accumulate > 1:
        trainer_kwargs['accumulate_grad_batches'] = accumulate

    trainer = pl.Trainer(**trainer_kwargs)

    # Save run info
    n_train = len(train_indices) if train_indices is not None else len(train_spectra)
    n_val = len(val_indices) if val_indices is not None else len(val_spectra)
    run_info_common = {
        "model_params": model.num_parameters,
        "embedding_type": config['model'].get('embedding_type', 'intensity'),
        "pretrain_loss": loss_type,
        "discretization": config.get('discretization'),
        "train_samples": n_train,
        "val_samples": n_val,
        "n_lines": config['data']['n_bins'] if use_line_token else None,
        "line_features_path": line_features_path,
        "line_tokens_path": line_tokens_path,
        "line_embedding_config": args.line_embedding_config,
        "libs_data_config": args.libs_data_config,
        "spectral_patch": spectral_patch_run_info(spectral_meta, model, args.libs_data_config),
        "batch_size": config['pretrain']['batch_size'],
        "epochs": config['pretrain']['epochs'],
        "mask_ratio": config['pretrain']['mask_ratio'],
        "contiguous_masking": contiguous_masking,
        "block_sizes": block_sizes,
        "peak_bias_enabled": peak_bias_enabled,
        "seed": args.seed,
    }
    run_mgr.save_run_info({**run_info_common, "status": "running"})

    # Train
    print("\nStarting pre-training...")
    print(f"Checkpoints: {run_mgr.checkpoint_dir}")
    print(f"Logs: {run_mgr.log_dir}")

    trainer.fit(pretrain_module, data_module)

    # Save final raw model weights
    final_path = run_mgr.checkpoint_dir / 'final_model.pt'
    torch.save(model.state_dict(), final_path)
    print(f"\nSaved final model to {final_path}")

    # Update run info
    n_train = len(train_indices) if train_indices is not None else len(train_spectra)
    n_val = len(val_indices) if val_indices is not None else len(val_spectra)
    run_mgr.save_run_info({
        **run_info_common,
        "status": "completed",
        "best_val_loss": float(checkpoint_callback.best_model_score) if checkpoint_callback.best_model_score else None,
        "final_model": str(final_path),
    })

    print("\n" + "="*60)
    print("Pre-training complete!")
    print("="*60)
    print(f"Run directory: {run_mgr.run_dir}")
    if checkpoint_callback.best_model_score:
        print(f"Best val loss: {checkpoint_callback.best_model_score:.4f}")
    if spectral_meta is None:  # evaluate_model.py handles the legacy intensity encoders only
        print(f"\nTo evaluate:")
        print(f"  uv run python evaluate_model.py --run_dir {run_mgr.run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-train LIBS Foundation Model")

    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to model/training config file')
    parser.add_argument('--libs_data_config', type=str, default=None,
                        help='Path to physics-based LIBS data pipeline config '
                             '(e.g. config/libs_data.yaml). If set, replaces the '
                             'legacy SyntheticLIBSGenerator with the realistic pipeline.')
    parser.add_argument('--line_embedding_config', type=str, default=None,
                        help='Path to config/line_embedding.yaml for line-as-token embedding.')
    parser.add_argument('--runs_dir', type=str, default='runs',
                        help='Base directory for all runs (default: runs/)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Optional experiment name (added to run folder name)')
    parser.add_argument('--save_data', action='store_true',
                        help='Save generated data to run directory')
    parser.add_argument('--contiguous_masking', action='store_true',
                        help='Use contiguous masking (overrides config)')
    parser.add_argument('--early_stopping', action='store_true',
                        help='Enable early stopping')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')

    args = parser.parse_args()
    main(args)
