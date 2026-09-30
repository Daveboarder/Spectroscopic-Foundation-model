"""
Pre-training module for LIBS Foundation Model.

Implements self-supervised pre-training using masked intensity prediction (MIP).
Supports regression (MSE) or classification (cross-entropy on discretized targets).
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Any, Dict, Optional, Tuple
import math
import numpy as np

from data.discretization import SpectroscopicDiscretizer


def compute_masked_classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    discretizer: SpectroscopicDiscretizer,
    criterion: nn.Module,
) -> torch.Tensor:
    """
    Cross-entropy on masked positions after on-the-fly discretization.

    Args:
        logits: ``[B, Seq, Num_Bins]`` or broadcastable after reshape.
        targets: Continuous targets, same leading shape as mask.
        mask: Boolean ``[B, Seq]`` (or broadcastable).
    """
    flat_mask = mask.reshape(-1)
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    masked_logits = flat_logits[flat_mask]
    masked_targets = flat_targets[flat_mask]
    if masked_logits.numel() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    class_targets = discretizer.to_bins(masked_targets)
    return criterion(masked_logits, class_targets)


def compute_masked_line_classification_loss(
    intensity_logits: torch.Tensor,
    fwhm_logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    intensity_discretizer: SpectroscopicDiscretizer,
    fwhm_discretizer: SpectroscopicDiscretizer,
    criterion: nn.Module,
) -> torch.Tensor:
    """CE on max_intensity (ch 0) and FWHM (ch 1) for line-token MIP."""
    loss_i = compute_masked_classification_loss(
        intensity_logits,
        targets[..., 0],
        mask,
        intensity_discretizer,
        criterion,
    )
    loss_f = compute_masked_classification_loss(
        fwhm_logits,
        targets[..., 1],
        mask,
        fwhm_discretizer,
        criterion,
    )
    return loss_i + loss_f


def _classification_decode_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    discretizer: SpectroscopicDiscretizer,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MAE, R², and bin accuracy on masked positions (decoded continuous preds)."""
    flat_mask = mask.reshape(-1)
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    masked_logits = flat_logits[flat_mask]
    masked_targets = flat_targets[flat_mask]
    if masked_logits.numel() == 0:
        z = torch.tensor(0.0, device=logits.device)
        return z, z, z
    pred_bins = masked_logits.argmax(dim=-1)
    pred_cont = discretizer.to_continuous(pred_bins)
    class_targets = discretizer.to_bins(masked_targets)
    bin_acc = (pred_bins == class_targets).float().mean()
    mae = torch.abs(pred_cont - masked_targets).mean()
    ss_res = ((masked_targets - pred_cont) ** 2).sum()
    ss_tot = ((masked_targets - masked_targets.mean()) ** 2).sum()
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    return mae, r2, bin_acc


class LIBSPretrainModule(pl.LightningModule):
    """
    PyTorch Lightning module for pre-training LIBS Transformer.
    
    Uses masked intensity prediction as the self-supervised objective.
    
    Args:
        model: LIBSTransformer model
        learning_rate: Initial learning rate
        weight_decay: Weight decay for AdamW
        warmup_epochs: Number of warmup epochs
        max_epochs: Maximum number of training epochs
        min_lr: Minimum learning rate for scheduler
        mask_ratio, mask_span_tokens: window masking of ``spectral_patch`` models,
            which is drawn here on the GPU (the other modes mask in the dataset)
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        min_lr: float = 1e-6,
        loss_type: str = "mse",
        intensity_discretizer: Optional[SpectroscopicDiscretizer] = None,
        fwhm_discretizer: Optional[SpectroscopicDiscretizer] = None,
        mask_ratio: float = 0.15,
        mask_span_tokens: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=['model', 'intensity_discretizer', 'fwhm_discretizer']
        )
        
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.loss_type = loss_type
        self.intensity_discretizer = intensity_discretizer
        self.fwhm_discretizer = fwhm_discretizer
        self.mask_ratio = float(mask_ratio)
        self.mask_span_tokens = int(mask_span_tokens)
        
        self.mse_loss = nn.MSELoss()
        self.ce_loss = (
            nn.CrossEntropyLoss(reduction='mean')
            if loss_type == "classification"
            else None
        )
    
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        if 'tokens' in batch:
            return self.model({'tokens': batch['tokens'], 'fit_valid': batch.get('fit_valid')})
        if 'line_features' in batch:
            return self.model({'line_features': batch['line_features']})
        x = batch.get('input', batch.get('spectrum'))
        return self.model(x, mask=mask)
    
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute masked intensity prediction loss.
        
        Args:
            predictions: Predicted intensities [batch_size, n_bins]
            targets: Target intensities [batch_size, n_bins]
            mask: Boolean mask [batch_size, n_bins]
            
        Returns:
            MSE loss over masked positions
        """
        # Select only masked positions
        masked_preds = predictions[mask]
        masked_targets = targets[mask]
        
        if masked_preds.numel() == 0:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        
        return self.mse_loss(masked_preds, masked_targets)

    def _is_patch_model(self) -> bool:
        return getattr(self.model, 'embedding_type', None) == 'spectral_patch'

    def _patch_step(
        self,
        batch: Dict[str, torch.Tensor],
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Masked-window reconstruction for ``spectral_patch`` models.

        Returns (loss, masked predictions, masked targets); predictions and
        targets are the asinh-transformed samples of the masked windows."""
        x = batch['spectrum']
        emb = self.model.embedding
        token_mask = emb.sample_token_mask(
            x.shape[0], self.mask_ratio, self.mask_span_tokens,
            device=x.device, generator=generator,
        )
        outputs = self.model(x, mask=token_mask)
        with torch.no_grad():
            targets = emb.window_targets(x)
        preds = outputs['mip_predictions'].float()
        m = token_mask.unsqueeze(-1).expand_as(preds)
        masked_preds, masked_targets = preds[m], targets[m]
        return self.mse_loss(masked_preds, masked_targets), masked_preds, masked_targets

    def _forward_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if 'tokens' in batch:
            return self.model({'tokens': batch['tokens'], 'fit_valid': batch.get('fit_valid')})
        if 'line_features' in batch:
            return self.model({'line_features': batch['line_features']})
        embedding_mask = (batch['mask_type'] == 1)
        return self.model(batch['input'], mask=embedding_mask)

    def _is_line_batch(self, batch: Dict[str, torch.Tensor]) -> bool:
        return 'tokens' in batch or 'line_features' in batch

    def _compute_batch_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if self.loss_type == "classification":
            line_mask = batch['mask']
            targets = batch['target']
            if 'fwhm_logits' in outputs:
                return compute_masked_line_classification_loss(
                    outputs['intensity_logits'],
                    outputs['fwhm_logits'],
                    targets,
                    line_mask,
                    self.intensity_discretizer,
                    self.fwhm_discretizer,
                    self.ce_loss,
                )
            return compute_masked_classification_loss(
                outputs['intensity_logits'],
                targets,
                line_mask if self._is_line_batch(batch) else batch['mask'],
                self.intensity_discretizer,
                self.ce_loss,
            )
        if self._is_line_batch(batch):
            return self.model.compute_mip_loss(
                outputs['mip_predictions'], batch['target'], batch['mask'],
            )
        return self.compute_loss(
            outputs['mip_predictions'], batch['target'], batch['mask'],
        )
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step.

        Args:
            batch: Dictionary with 'input', 'target', 'mask', 'mask_type'
            batch_idx: Batch index

        Returns:
            Loss value
        """
        if self._is_patch_model():
            loss, _, _ = self._patch_step(batch)
        else:
            outputs = self._forward_batch(batch)
            loss = self._compute_batch_loss(outputs, batch)
        
        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/lr', self.optimizers().param_groups[0]['lr'], on_step=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Validation step.

        Args:
            batch: Dictionary with 'input', 'target', 'mask', 'mask_type'
            batch_idx: Batch index

        Returns:
            Dictionary with loss and predictions
        """
        if self._is_patch_model():
            # fixed masks per batch index, so val/loss is comparable across epochs
            gen = torch.Generator(device=batch['spectrum'].device)
            gen.manual_seed(1_000_003 + batch_idx)
            loss, masked_preds, masked_targets = self._patch_step(batch, generator=gen)
            with torch.no_grad():
                mae = torch.abs(masked_preds - masked_targets).mean()
                ss_res = ((masked_targets - masked_preds) ** 2).sum()
                ss_tot = ((masked_targets - masked_targets.mean()) ** 2).sum()
                r2 = 1 - ss_res / (ss_tot + 1e-8)
            self.log('val/loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('val/mae', mae, on_epoch=True, sync_dist=True)
            self.log('val/r2', r2, on_epoch=True, sync_dist=True)
            return {'loss': loss, 'predictions': None}
        outputs = self._forward_batch(batch)
        loss = self._compute_batch_loss(outputs, batch)
        line_mask = batch['mask']
        targets = batch['target']

        with torch.no_grad():
            if self.loss_type == "classification":
                if 'fwhm_logits' in outputs:
                    mae_i, r2_i, acc_i = _classification_decode_metrics(
                        outputs['intensity_logits'],
                        targets[..., 0],
                        line_mask,
                        self.intensity_discretizer,
                    )
                    mae_f, r2_f, acc_f = _classification_decode_metrics(
                        outputs['fwhm_logits'],
                        targets[..., 1],
                        line_mask,
                        self.fwhm_discretizer,
                    )
                    mae = (mae_i + mae_f) * 0.5
                    r2 = (r2_i + r2_f) * 0.5
                    bin_acc = (acc_i + acc_f) * 0.5
                else:
                    mae, r2, bin_acc = _classification_decode_metrics(
                        outputs['intensity_logits'],
                        targets,
                        line_mask if self._is_line_batch(batch) else batch['mask'],
                        self.intensity_discretizer,
                    )
            else:
                bin_acc = torch.tensor(0.0, device=loss.device)
                if self._is_line_batch(batch):
                    preds = outputs['mip_predictions']
                    m = line_mask.unsqueeze(-1).expand_as(preds)
                    masked_preds = preds[m]
                    masked_targets = targets[m]
                else:
                    masked_preds = outputs['mip_predictions'][batch['mask']]
                    masked_targets = targets[batch['mask']]
                if masked_preds.numel() > 0:
                    mae = torch.abs(masked_preds - masked_targets).mean()
                    ss_res = ((masked_targets - masked_preds) ** 2).sum()
                    ss_tot = ((masked_targets - masked_targets.mean()) ** 2).sum()
                    r2 = 1 - ss_res / (ss_tot + 1e-8)
                else:
                    mae = torch.tensor(0.0, device=loss.device)
                    r2 = torch.tensor(0.0, device=loss.device)
        
        # Log metrics
        self.log('val/loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/mae', mae, on_epoch=True, sync_dist=True)
        self.log('val/r2', r2, on_epoch=True, sync_dist=True)
        if self.loss_type == "classification":
            self.log('val/bin_accuracy', bin_acc, on_epoch=True, sync_dist=True)
            if 'fwhm_logits' in outputs:
                self.log('val/mae_intensity', mae_i, on_epoch=True, sync_dist=True)
                self.log('val/mae_fwhm', mae_f, on_epoch=True, sync_dist=True)
        
        pred_key = (
            'intensity_logits' if self.loss_type == "classification"
            else 'mip_predictions'
        )
        return {'loss': loss, 'predictions': outputs.get(pred_key)}
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # AdamW optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )
        
        # Cosine annealing with warmup
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                # Linear warmup
                return (epoch + 1) / self.warmup_epochs
            else:
                # Cosine annealing
                progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                min_factor = self.min_lr / self.learning_rate
                return min_factor + (1 - min_factor) * cosine_decay
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            },
        }
    
    def on_train_epoch_end(self):
        """Called at the end of each training epoch."""
        # Log current epoch
        self.log('epoch', float(self.current_epoch))


class PretrainDataModule(pl.LightningDataModule):
    """
    DataModule for pre-training.
    
    Handles data loading for self-supervised pre-training with advanced masking.
    
    Args:
        train_spectra: Training spectra array
        val_spectra: Validation spectra array
        batch_size: Batch size
        mask_ratio: Fraction of positions to mask
        contiguous_masking: Whether to use contiguous block masking
        block_sizes: List of possible block sizes (e.g., [25, 50, 100, 150])
        peak_bias_enabled: Whether to bias masking toward peaks
        peak_bias_ratio: Fraction of masks that should cover peaks
        peak_threshold: Intensity threshold for peak detection
        num_workers: Number of data loading workers
        
        # Legacy parameter (backward compatibility)
        contiguous_mask_size: Single block size (overridden by block_sizes if provided)
    """
    
    def __init__(
        self,
        train_spectra=None,
        val_spectra=None,
        batch_size: int = 64,
        mask_ratio: float = 0.15,
        contiguous_masking: bool = False,
        block_sizes: Optional[list] = None,
        peak_bias_enabled: bool = False,
        peak_bias_ratio: float = 0.5,
        peak_threshold: float = 0.2,
        num_workers: int = 0,
        contiguous_mask_size: int = 50,
        line_features_path: Optional[str] = None,
        line_tokens_path: Optional[str] = None,
        train_indices: Optional[np.ndarray] = None,
        val_indices: Optional[np.ndarray] = None,
        spectral_patch: bool = False,
    ):
        super().__init__()
        self.spectral_patch = bool(spectral_patch)
        self.train_spectra = train_spectra
        self.val_spectra = val_spectra
        self.batch_size = batch_size
        self.mask_ratio = mask_ratio
        self.contiguous_masking = contiguous_masking
        self.block_sizes = block_sizes if block_sizes is not None else [contiguous_mask_size]
        self.peak_bias_enabled = peak_bias_enabled
        self.peak_bias_ratio = peak_bias_ratio
        self.peak_threshold = peak_threshold
        self.num_workers = num_workers
        self.line_features_path = line_features_path
        self.line_tokens_path = line_tokens_path
        self.train_indices = train_indices
        self.val_indices = val_indices
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets."""
        from data.dataset import (
            MaskedLIBSDataset,
            MaskedLineTokenDataset,
            MaskedLineTokensDataset,
            SpectrumDataset,
        )
        import numpy as np
        
        if stage == 'fit' or stage is None:
            if self.spectral_patch:
                # raw spectra; the window masks are drawn by LIBSPretrainModule
                self.train_dataset = SpectrumDataset(self.train_spectra)
                self.val_dataset = SpectrumDataset(self.val_spectra)
                print(f"\nSpectral-patch pretrain (masked windows): "
                      f"{len(self.train_dataset)} train / {len(self.val_dataset)} val spectra")
            elif self.line_tokens_path:
                self.train_dataset = MaskedLineTokensDataset(
                    self.line_tokens_path,
                    indices=self.train_indices,
                    mask_ratio=self.mask_ratio,
                )
                self.val_dataset = MaskedLineTokensDataset(
                    self.line_tokens_path,
                    indices=self.val_indices,
                    mask_ratio=self.mask_ratio,
                    seed=43,
                )
                print(f"\nLine-tokens pretrain (pre-baked HDF5): "
                      f"{len(self.train_dataset)} train / {len(self.val_dataset)} val spectra")
            elif self.line_features_path:
                self.train_dataset = MaskedLineTokenDataset(
                    self.line_features_path,
                    indices=self.train_indices,
                    mask_ratio=self.mask_ratio,
                )
                self.val_dataset = MaskedLineTokenDataset(
                    self.line_features_path,
                    indices=self.val_indices,
                    mask_ratio=self.mask_ratio,
                    seed=43,
                )
                print(f"\nLine-token pretrain: {len(self.train_dataset)} train / "
                      f"{len(self.val_dataset)} val spectra")
            else:
                self.train_dataset = MaskedLIBSDataset(
                    spectra=self.train_spectra,
                    mask_ratio=self.mask_ratio,
                    contiguous_masking=self.contiguous_masking,
                    block_sizes=self.block_sizes,
                    peak_bias_enabled=self.peak_bias_enabled,
                    peak_bias_ratio=self.peak_bias_ratio,
                    peak_threshold=self.peak_threshold,
                )
                self.val_dataset = MaskedLIBSDataset(
                    spectra=self.val_spectra,
                    mask_ratio=self.mask_ratio,
                    contiguous_masking=self.contiguous_masking,
                    block_sizes=self.block_sizes,
                    peak_bias_enabled=self.peak_bias_enabled,
                    peak_bias_ratio=self.peak_bias_ratio,
                    peak_threshold=self.peak_threshold,
                )
                stats = self.train_dataset.get_masking_stats(n_samples=100)
                print(f"\nMasking Statistics (from 100 samples):")
                print(f"  Avg masked bins: {stats['avg_masked_bins']:.1f}")
                print(f"  Avg masked ratio: {stats['avg_masked_ratio']:.2%}")
                print(f"  Peak coverage: {stats['peak_coverage']:.2%}")
                print(f"  Avg peak bins: {stats['avg_peak_bins']:.1f}")
    
    def train_dataloader(self):
        from data.dataset import libs_worker_init_fn
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=libs_worker_init_fn if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        from data.dataset import libs_worker_init_fn
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=libs_worker_init_fn if self.num_workers > 0 else None,
        )


if __name__ == "__main__":
    import sys
    sys.path.append('.')
    
    from models.libs_transformer import LIBSTransformer
    from data.synthetic_generator import SyntheticLIBSGenerator
    from data.dataset import MaskedLIBSDataset
    
    # Generate synthetic data
    print("Generating synthetic data...")
    generator = SyntheticLIBSGenerator(seed=42)
    spectra, _, _ = generator.generate_dataset(n_samples=1000)
    
    # Split data
    train_spectra = spectra[:800]
    val_spectra = spectra[800:]
    
    # Create model
    print("Creating model...")
    model = LIBSTransformer(
        n_bins=2048,
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        dropout=0.1,
    )
    
    # Create Lightning module
    pretrain_module = LIBSPretrainModule(
        model=model,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_epochs=2,
        max_epochs=10,
    )
    
    print(f"Model parameters: {model.num_parameters:,}")
    
    # Create data module
    data_module = PretrainDataModule(
        train_spectra=train_spectra,
        val_spectra=val_spectra,
        batch_size=32,
        mask_ratio=0.15,
    )
    
    # Test a single batch
    print("\nTesting single batch...")
    data_module.setup('fit')
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    
    print(f"Batch keys: {batch.keys()}")
    print(f"Input shape: {batch['input'].shape}")
    print(f"Target shape: {batch['target'].shape}")
    print(f"Mask shape: {batch['mask'].shape}")
    
    # Test training step
    loss = pretrain_module.training_step(batch, 0)
    print(f"Training loss: {loss.item():.4f}")
