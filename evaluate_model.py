"""
Evaluation script for LIBS Foundation Model.

Generates comprehensive visualizations and metrics after training:
- Pre-training: MIP reconstruction quality, error analysis, t-SNE embeddings
- Fine-tuning: Classification metrics, confusion matrix, regression scatter, t-SNE

Usage:
    python evaluate_model.py --run_dir runs/pretrain_2024-02-05_14-30-25_my_exp
    python evaluate_model.py --run_dir runs/finetune_2024-02-06_09-00-00_both

Outputs are saved to: {run_dir}/evaluation/eval_{timestamp}/
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).parent))

from data.synthetic_generator import SyntheticLIBSGenerator, MATERIAL_CLASSES
from data.dataset import MaskedLIBSDataset, LabeledLIBSDataset
from data.discretization import build_discretizer_from_config
from models.libs_transformer import LIBSTransformer
from models.heads import ClassificationHead, RegressionHead
from utils.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
    compute_mip_metrics,
    format_metrics,
    plot_confusion_matrix,
    plot_spectra_comparison,
    plot_embeddings_tsne,
    plot_regression_scatter,
)
from utils.run_manager import RunManager
import yaml


def _infer_embedding_type_from_state_dict(state_dict: dict, fallback: str = "intensity") -> str:
    """Infer encoder embedding type from checkpoint keys."""
    if any(k.startswith("embedding.projection.") for k in state_dict.keys()):
        return "line_token_linear"
    if any(k.startswith("embedding.quantum_proj.") for k in state_dict.keys()):
        return "line_token"
    if any(k.startswith("embedding.intensity_projection.") for k in state_dict.keys()):
        return "intensity"
    if any(k.startswith("embedding.patch_proj.") for k in state_dict.keys()):
        return "spectral_patch"
    return fallback


def _line_token_meta_from_encoder_state(encoder_state: dict) -> dict:
    """Build minimal line_token_linear meta from checkpoint buffers."""
    mean = encoder_state.get("embedding.feature_mean")
    std = encoder_state.get("embedding.feature_std")
    wl = encoder_state.get("embedding.central_wavelength")
    if mean is None or std is None or wl is None:
        raise ValueError(
            "line_token_linear checkpoint is missing embedding buffers "
            "(feature_mean, feature_std, central_wavelength)."
        )
    return {
        "n_features": int(mean.numel()),
        "feature_mean": mean.cpu().numpy(),
        "feature_std": std.cpu().numpy(),
        "central_wavelength": wl.cpu().numpy(),
        "n_lines": int(wl.numel()),
    }


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _create_model(
    config: dict,
    embedding_type: str | None = None,
    line_token_meta: dict | None = None,
) -> LIBSTransformer:
    """Create a LIBSTransformer from config (with dropout=0 for evaluation)."""
    emb = embedding_type or config['model'].get('embedding_type', 'intensity')
    n_mip_channels = int(config['model'].get('n_mip_target_channels', 2))
    n_bins = int(config['data']['n_bins'])
    n_lines = int(config['data'].get('n_lines', n_bins))
    if emb == "line_token_linear" and line_token_meta is not None:
        n_lines = int(line_token_meta["n_lines"])
        n_bins = n_lines
    loss_type = str(config.get('pretrain', {}).get('loss', 'mse')).lower()
    return LIBSTransformer(
        n_bins=n_bins,
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        n_layers=config['model']['n_layers'],
        d_ff=config['model']['d_ff'],
        dropout=0.0,
        n_classes=config['data']['n_classes'],
        embedding_type=emb,
        n_lines=n_lines,
        line_token_meta=line_token_meta,
        n_mip_target_channels=n_mip_channels,
        mip_loss_type=loss_type,
        num_intensity_bins=int(config['model'].get('num_intensity_bins', 256)),
        num_fwhm_bins=int(config['model'].get('num_fwhm_bins', 100)),
    )


def load_pretrain_model(checkpoint_path: str, config: dict) -> LIBSTransformer:
    """
    Load a pre-trained model from checkpoint.

    Handles:
    - Raw state dict (.pt) saved via torch.save(model.state_dict(), ...)
    - Lightning checkpoint (.ckpt) where keys are prefixed with 'model.'
    """
    model = _create_model(config)

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # Pretrain Lightning module stores encoder as 'model.*'
        if any(k.startswith('model.') for k in state_dict.keys()):
            state_dict = {k[len('model.'):]: v for k, v in state_dict.items()
                          if k.startswith('model.')}
        print(f"Loaded Lightning checkpoint (epoch {checkpoint.get('epoch', '?')})")
    else:
        state_dict = checkpoint
        print("Loaded raw model weights")

    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_finetune_model(checkpoint_path: str, config: dict):
    """
    Load a fine-tuned model from a Lightning checkpoint.

    Returns (model, cls_head, reg_head).
    The Lightning module uses attributes:
        self.encoder  -> LIBSTransformer
        self.classification_head -> ClassificationHead
        self.regression_head -> RegressionHead
    So keys in the checkpoint are: encoder.*, classification_head.*, regression_head.*
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Recover pool from Lightning hparams so heads are sized correctly.
    # Older checkpoints predate the pool option → default to 'cls'.
    pool = 'cls'
    if isinstance(checkpoint, dict):
        pool = checkpoint.get('hyper_parameters', {}).get('pool', 'cls')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        epoch = checkpoint.get('epoch', '?')
        print(f"Loaded Lightning checkpoint (epoch {epoch})")
    else:
        state_dict = checkpoint

    def _extract(prefix: str) -> dict:
        return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

    encoder_state = _extract('encoder.')
    cls_state = _extract('classification_head.')
    reg_state = _extract('regression_head.')

    emb_fallback = config['model'].get('embedding_type', 'intensity')
    emb_type = _infer_embedding_type_from_state_dict(encoder_state, fallback=emb_fallback)
    line_token_meta = _line_token_meta_from_encoder_state(encoder_state) if emb_type == "line_token_linear" else None
    model = _create_model(config, embedding_type=emb_type, line_token_meta=line_token_meta)
    d_model = config['model']['d_model']
    n_classes = config['data']['n_classes']
    head_in_dim = 2 * d_model if pool == 'cls_mean' else d_model
    print(f"Finetune pool: {pool} (head input dim={head_in_dim})")
    print(f"Encoder embedding_type inferred as: {emb_type}")

    if encoder_state:
        # Finetune evaluation does not need pretrain MIP head; skip it to avoid
        # shape mismatches across embedding variants / target-channel counts.
        encoder_state = {k: v for k, v in encoder_state.items() if not k.startswith("mip_head.")}
        model.load_state_dict(encoder_state, strict=False)
    else:
        print("WARNING: no encoder weights found in checkpoint")

    cls_head = ClassificationHead(d_model=head_in_dim, n_classes=n_classes)
    reg_head = RegressionHead(d_model=head_in_dim, n_outputs=n_classes)

    if cls_state:
        cls_head.load_state_dict(cls_state)
    else:
        print("WARNING: no classification_head weights found (may be regression-only)")

    if reg_state:
        reg_head.load_state_dict(reg_state)
    else:
        print("WARNING: no regression_head weights found (may be classification-only)")

    model.eval()
    cls_head.eval()
    reg_head.eval()
    return model, cls_head, reg_head, pool


def generate_test_data(config: dict, n_samples: int = 1000, seed: int = 9999):
    """Generate held-out test data with a different seed."""
    generator = SyntheticLIBSGenerator(
        n_bins=config['data']['n_bins'],
        noise_sigma=config['data']['synthetic']['noise_sigma'],
        peak_width_range=tuple(config['data']['synthetic']['peak_width_range']),
        intensity_variation=config['data']['synthetic']['intensity_variation'],
        seed=seed,
    )

    spectra, labels, concentrations = generator.generate_dataset(
        n_samples=n_samples,
        return_labels=True,
    )

    return spectra, labels, concentrations


# ────────────────────────────────────────────────────────────────────────────
# Pre-train evaluation
# ────────────────────────────────────────────────────────────────────────────

def evaluate_pretrain(
    model: LIBSTransformer,
    config: dict,
    output_dir: Path,
    device: str = 'cuda',
):
    print("\n" + "="*60)
    print("EVALUATING PRE-TRAINED MODEL (MIP)")
    print("="*60)

    print("\nGenerating test data...")
    spectra, labels, concentrations = generate_test_data(config, n_samples=1000)

    # Create masked dataset
    mask_dataset = MaskedLIBSDataset(
        spectra=spectra,
        mask_ratio=config['pretrain']['mask_ratio'],
        contiguous_masking=config['pretrain'].get('contiguous_masking', False),
        block_sizes=config['pretrain'].get('block_sizes', [50]),
        peak_bias_enabled=config['pretrain'].get('peak_bias_enabled', False),
        peak_bias_ratio=config['pretrain'].get('peak_bias_ratio', 0.5),
        peak_threshold=config['pretrain'].get('peak_threshold', 0.2),
        seed=42,
    )

    model = model.to(device)

    # Build the discretizer once (classification MIP only) — hoisted out of the
    # batch loop so logspace boundaries are not recomputed per step.
    loss_type = str(config.get('pretrain', {}).get('loss', 'mse')).lower()
    discretizer = None
    if loss_type == 'classification':
        disc_cfg = config.get('discretization', {}).get('intensity') or {
            'num_bins': config['model'].get('num_intensity_bins', 256),
            'min_val': 1.0e-4,
            'max_val': 1.0,
            'strategy': 'log',
        }
        discretizer = build_discretizer_from_config(disc_cfg).to(device)

    print("Running inference...")
    all_preds = []
    all_targets = []
    all_masks = []
    all_inputs = []

    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(mask_dataset), batch_size):
            batch_items = [mask_dataset[j] for j in range(i, min(i + batch_size, len(mask_dataset)))]

            inputs = torch.stack([item['input'] for item in batch_items]).to(device)
            masks = torch.stack([item['mask'] for item in batch_items])
            mask_types = torch.stack([item['mask_type'] for item in batch_items])
            targets = torch.stack([item['target'] for item in batch_items])

            # Match training: only mask-token positions (type 1) get mask embedding
            embedding_mask = (mask_types == 1).to(device)
            output = model(inputs, mask=embedding_mask)
            if discretizer is not None:
                pred_bins = output['intensity_logits'].argmax(dim=-1)
                predictions = discretizer.to_continuous(pred_bins).cpu()
            else:
                predictions = output['mip_predictions'].cpu()

            all_preds.append(predictions.numpy())
            all_targets.append(targets.numpy())
            all_masks.append(masks.numpy())
            all_inputs.append(inputs.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_masks = np.concatenate(all_masks, axis=0).astype(bool)
    all_inputs = np.concatenate(all_inputs, axis=0)

    # Compute metrics (masked positions only)
    mip_metrics = compute_mip_metrics(all_preds, all_targets, all_masks)

    print("\nMIP Metrics (masked positions only):")
    print(format_metrics(mip_metrics))

    # ── Figure 1: Reconstruction examples ──
    print("\nGenerating visualizations...")
    n_examples = 6
    fig, axes = plt.subplots(n_examples, 1, figsize=(14, 3 * n_examples))
    fig.suptitle("MIP Reconstruction Examples (Masked Regions Highlighted)", fontsize=14, fontweight='bold')

    for idx in range(n_examples):
        ax = axes[idx]
        wavelengths = np.arange(config['data']['n_bins'])

        ax.plot(wavelengths, all_targets[idx], 'b-', alpha=0.7, linewidth=0.8, label='Original')
        ax.plot(wavelengths, all_preds[idx], 'r-', alpha=0.7, linewidth=0.8, label='Predicted')

        mask_regions = all_masks[idx]
        ax.fill_between(wavelengths, 0, 1, where=mask_regions,
                         alpha=0.15, color='yellow', label='Masked Region')

        class_name = MATERIAL_CLASSES[labels[idx]].name if idx < len(labels) else "?"
        ax.set_title(f"Sample {idx} — Class: {class_name}", fontsize=10)
        ax.set_xlim(0, config['data']['n_bins'])
        ax.set_ylim(-0.05, 1.05)
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)
        if idx == n_examples - 1:
            ax.set_xlabel("Wavelength Bin")
        ax.set_ylabel("Intensity")

    plt.tight_layout()
    plt.savefig(output_dir / "pretrain_reconstruction_examples.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: pretrain_reconstruction_examples.png")

    # ── Figure 2: Error distribution ──
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("MIP Error Analysis", fontsize=14, fontweight='bold')

    errors = (all_preds - all_targets)[all_masks]
    axes[0, 0].hist(errors, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel("Prediction Error")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title(f"Error Distribution (Masked Only)\nMean: {errors.mean():.4f}, Std: {errors.std():.4f}")

    true_intensities = all_targets[all_masks]
    abs_errors = np.abs(errors)

    intensity_bins = np.linspace(0, 1, 11)
    bin_indices = np.digitize(true_intensities, intensity_bins)
    mean_errors = [abs_errors[bin_indices == i].mean() if (bin_indices == i).sum() > 0 else 0
                   for i in range(1, len(intensity_bins))]

    axes[0, 1].bar(range(len(mean_errors)), mean_errors, color='coral', edgecolor='black')
    axes[0, 1].set_xticks(range(len(mean_errors)))
    axes[0, 1].set_xticklabels([f"{intensity_bins[i]:.1f}-{intensity_bins[i+1]:.1f}"
                                for i in range(len(intensity_bins)-1)], rotation=45)
    axes[0, 1].set_xlabel("True Intensity Range")
    axes[0, 1].set_ylabel("Mean Absolute Error")
    axes[0, 1].set_title("Error by Intensity Level")

    sample_size = min(5000, len(true_intensities))
    sample_idx = np.random.choice(len(true_intensities), sample_size, replace=False)
    axes[1, 0].scatter(true_intensities[sample_idx], all_preds[all_masks][sample_idx],
                       alpha=0.3, s=1, c='steelblue')
    axes[1, 0].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect')
    axes[1, 0].set_xlabel("True Intensity")
    axes[1, 0].set_ylabel("Predicted Intensity")
    axes[1, 0].set_title(f"Predicted vs True (R² = {mip_metrics['r2']:.4f})")
    axes[1, 0].legend()
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)

    class_errors = []
    class_names = [MATERIAL_CLASSES[i].name for i in range(5)]
    for class_id in range(5):
        class_mask = labels == class_id
        if class_mask.sum() > 0:
            class_abs_errors = np.abs(all_preds[class_mask] - all_targets[class_mask])[all_masks[class_mask]]
            class_errors.append(class_abs_errors.mean())
        else:
            class_errors.append(0)

    axes[1, 1].bar(class_names, class_errors, color=plt.cm.tab10(np.arange(5)/10), edgecolor='black')
    axes[1, 1].set_xlabel("Material Class")
    axes[1, 1].set_ylabel("Mean Absolute Error")
    axes[1, 1].set_title("Reconstruction Error by Class")
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / "pretrain_error_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: pretrain_error_analysis.png")

    # ── Figure 3: Embedding visualization ──
    print("Computing embeddings for t-SNE...")
    embeddings = []
    with torch.no_grad():
        for i in range(min(500, len(spectra))):
            spec = torch.tensor(spectra[i:i+1], dtype=torch.float32).to(device)
            output = model(spec)
            embeddings.append(output['cls_embedding'].cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)

    fig, ax = plt.subplots(figsize=(10, 8))
    plot_embeddings_tsne(
        embeddings,
        labels[:len(embeddings)],
        class_names=[MATERIAL_CLASSES[i].name for i in range(5)],
        title="t-SNE of CLS Embeddings (Pre-trained Model)",
        ax=ax,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "pretrain_embeddings_tsne.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: pretrain_embeddings_tsne.png")

    # Save metrics
    with open(output_dir / "pretrain_metrics.txt", 'w') as f:
        f.write("Pre-training Evaluation Metrics\n")
        f.write("="*40 + "\n\n")
        f.write(format_metrics(mip_metrics))
    print("Saved: pretrain_metrics.txt")

    return mip_metrics


# ────────────────────────────────────────────────────────────────────────────
# Fine-tune evaluation
# ────────────────────────────────────────────────────────────────────────────

def evaluate_finetune(
    checkpoint_path: str,
    config: dict,
    output_dir: Path,
    device: str = 'cuda',
):
    print("\n" + "="*60)
    print("EVALUATING FINE-TUNED MODEL")
    print("="*60)

    model, cls_head, reg_head, pool = load_finetune_model(checkpoint_path, config)
    model = model.to(device)
    cls_head = cls_head.to(device)
    reg_head = reg_head.to(device)

    if getattr(model, "embedding_type", "intensity") != "intensity":
        raise NotImplementedError(
            "evaluate_model.py finetune path currently evaluates synthetic bin spectra only. "
            f"Loaded encoder embedding_type='{model.embedding_type}', which requires line-token "
            "inputs instead of [B, n_bins] spectra."
        )

    print("\nGenerating test data...")
    spectra, labels, concentrations = generate_test_data(config, n_samples=1000)

    print("Running inference...")
    all_embeddings = []
    all_cls_preds = []
    all_reg_preds = []

    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(spectra), batch_size):
            batch = torch.tensor(spectra[i:i+batch_size], dtype=torch.float32).to(device)

            output = model(batch)
            cls_emb = output['cls_embedding']

            # Apply the same pooling used at fine-tune time
            if pool == 'cls':
                rep = cls_emb
            elif pool == 'mean':
                rep = output['sequence_embeddings'].mean(dim=1)
            else:  # cls_mean
                rep = torch.cat([cls_emb, output['sequence_embeddings'].mean(dim=1)], dim=-1)

            cls_logits = cls_head(rep)
            reg_preds = reg_head(rep)

            # t-SNE on CLS embedding (kept consistent across pool choices)
            all_embeddings.append(cls_emb.cpu().numpy())
            all_cls_preds.append(cls_logits.cpu().numpy())
            all_reg_preds.append(reg_preds.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    cls_preds = np.concatenate(all_cls_preds, axis=0)
    reg_preds = np.concatenate(all_reg_preds, axis=0)

    pred_labels = cls_preds.argmax(axis=1)

    # Compute metrics
    print("\nComputing metrics...")
    class_names = [MATERIAL_CLASSES[i].name for i in range(5)]

    cls_metrics = compute_classification_metrics(
        pred_labels, labels,
        n_classes=5,
        class_names=class_names,
    )

    reg_metrics = compute_regression_metrics(
        reg_preds, concentrations,
        n_outputs=5,
        output_names=class_names,
    )

    print("\nClassification Metrics:")
    print(format_metrics({k: v for k, v in cls_metrics.items() if k != 'confusion_matrix'}))

    print("\nRegression Metrics:")
    print(format_metrics(reg_metrics))

    # ── Figure 1: Classification results ──
    print("\nGenerating visualizations...")
    fig = plt.figure(figsize=(16, 6))
    fig.suptitle("Classification Results", fontsize=14, fontweight='bold')

    gs = GridSpec(1, 3, figure=fig, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    plot_confusion_matrix(
        cls_metrics['confusion_matrix'],
        class_names=class_names,
        title=f"Confusion Matrix\nAccuracy: {cls_metrics['accuracy']:.1%}",
        ax=ax1,
    )

    ax2 = fig.add_subplot(gs[0, 1])
    per_class_acc = cls_metrics['confusion_matrix'].diagonal() / cls_metrics['confusion_matrix'].sum(axis=1).clip(1)
    bars = ax2.bar(class_names, per_class_acc, color=plt.cm.tab10(np.arange(5)/10), edgecolor='black')
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Per-Class Accuracy")
    ax2.tick_params(axis='x', rotation=45)
    for bar, acc in zip(bars, per_class_acc):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                 f'{acc:.1%}', ha='center', va='bottom', fontsize=9)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    metrics_text = (
        f"Overall Accuracy: {cls_metrics['accuracy']:.1%}\n"
        f"Macro F1: {cls_metrics.get('f1_macro', 0):.4f}\n"
        f"Weighted F1: {cls_metrics.get('f1_weighted', 0):.4f}"
    )
    ax3.text(0.5, 0.5, metrics_text, transform=ax3.transAxes,
             fontsize=14, verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / "finetune_classification.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: finetune_classification.png")

    # ── Figure 2: Regression results ──
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Concentration Regression Results", fontsize=14, fontweight='bold')

    for idx, name in enumerate(class_names):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]
        plot_regression_scatter(
            reg_preds[:, idx], concentrations[:, idx],
            title=f"{name} Concentration",
            ax=ax,
        )

    axes[1, 2].axis('off')
    metrics_text = f"Overall MAE: {reg_metrics.get('mae', 0):.4f}\nOverall R²: {reg_metrics.get('r2', 0):.4f}"
    axes[1, 2].text(0.5, 0.5, metrics_text, transform=axes[1, 2].transAxes,
                    fontsize=12, verticalalignment='center', horizontalalignment='center',
                    fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / "finetune_regression.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: finetune_regression.png")

    # ── Figure 3: t-SNE ──
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_embeddings_tsne(
        embeddings,
        labels,
        class_names=class_names,
        title="t-SNE of CLS Embeddings (Fine-tuned Model)",
        ax=ax,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "finetune_embeddings_tsne.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: finetune_embeddings_tsne.png")

    # Save metrics
    with open(output_dir / "finetune_metrics.txt", 'w') as f:
        f.write("Fine-tuning Evaluation Metrics\n")
        f.write("="*40 + "\n\n")
        f.write("CLASSIFICATION\n")
        f.write("-"*20 + "\n")
        f.write(format_metrics({k: v for k, v in cls_metrics.items() if k != 'confusion_matrix'}))
        f.write("\n\nREGRESSION\n")
        f.write("-"*20 + "\n")
        f.write(format_metrics(reg_metrics))
    print("Saved: finetune_metrics.txt")

    return cls_metrics, reg_metrics


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main(args):
    run_mgr = RunManager.from_existing_run(args.run_dir)

    # Detect mode from run type
    mode = args.mode or ("finetune" if "finetune" in run_mgr.run_type else "pretrain")

    # Pick the right checkpoint for this mode
    checkpoint_path = run_mgr.get_checkpoint_for_mode(mode)
    if checkpoint_path is None:
        raise ValueError(f"No checkpoint found in {run_mgr.checkpoint_dir}")

    # Load config from run directory
    config_path = run_mgr.run_dir / "config.yaml"
    if not config_path.exists():
        raise ValueError(f"Config not found in run directory: {config_path}")
    config = load_config(str(config_path))
    if config.get("model", {}).get("embedding_type") == "spectral_patch":
        raise NotImplementedError(
            "evaluate_model.py scores legacy 2048-bin intensity encoders only; evaluate "
            "spectral_patch runs from run_info.yaml test_results or with "
            "scripts/predict_mineral_map.py"
        )

    # Output to evaluation subfolder with timestamp
    eval_name = datetime.now().strftime("eval_%Y-%m-%d_%H-%M-%S")
    output_dir = run_mgr.get_evaluation_dir(eval_name)

    print(f"\n{'='*60}")
    print(f"Evaluating run: {run_mgr.run_name}")
    print(f"{'='*60}")
    print(f"Mode: {mode}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if mode == 'pretrain':
        model = load_pretrain_model(str(checkpoint_path), config)
        evaluate_pretrain(model, config, output_dir, device)
    elif mode == 'finetune':
        evaluate_finetune(str(checkpoint_path), config, output_dir, device)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    print("\n" + "="*60)
    print(f"Evaluation complete! Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate LIBS Foundation Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_model.py --run_dir runs/pretrain_2024-02-05_14-30-25_my_exp
  python evaluate_model.py --run_dir runs/finetune_2024-02-06_... --mode finetune
        """
    )

    parser.add_argument(
        '--run_dir',
        type=str,
        required=True,
        help='Path to run directory (contains checkpoints/, config.yaml)',
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['pretrain', 'finetune'],
        default=None,
        help='Evaluation mode (auto-detected from run_dir if not specified)',
    )

    args = parser.parse_args()
    main(args)
