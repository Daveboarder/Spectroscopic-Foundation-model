"""
LIBS Transformer Model.

Main transformer encoder architecture for LIBS spectral analysis.
Adapted for continuous intensity values with masked intensity prediction.
"""

import math
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union

from .positional_encoding import SinusoidalPositionalEncoding
from .line_token_embedding import LineTokenEmbedding, LinearLineTokenEmbedding
from .spectral_patch_embedding import SpectralPatchEmbedding
from .heads import MaskedBinIntensityHead, MaskedLineFeatureHead


class SpectralEmbedding(nn.Module):
    """
    Embedding layer for LIBS spectra.
    
    Projects each intensity value to the model dimension and adds
    positional encoding for wavelength positions.
    
    Args:
        d_model: Model dimension
        n_bins: Number of wavelength bins
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        d_model: int = 256,
        n_bins: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_bins = n_bins
        
        # Project each intensity value (scalar) to d_model dimensions
        self.intensity_projection = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )
        
        # Learnable CLS token for sequence-level representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Learnable MASK token for masked positions
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_len=n_bins + 1,  # +1 for CLS token
            dropout=dropout,
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Embed spectrum and add positional encoding.
        
        Args:
            x: Input spectrum of shape [batch_size, n_bins]
            mask: Optional boolean mask of shape [batch_size, n_bins]
                  indicating positions to mask (True = masked)
        
        Returns:
            Embedded sequence of shape [batch_size, n_bins + 1, d_model]
        """
        batch_size = x.shape[0]
        
        # Expand intensity to [batch, n_bins, 1]
        x = x.unsqueeze(-1)
        
        # Project intensities: [batch, n_bins, d_model]
        x = self.intensity_projection(x)
        
        # Apply mask token to masked positions if mask is provided
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(x)
            mask_token_expanded = self.mask_token.expand(batch_size, self.n_bins, -1)
            x = torch.where(mask_expanded, mask_token_expanded, x)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [batch, n_bins + 1, d_model]
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        return x


class TransformerEncoderBlock(nn.Module):
    """
    Single transformer encoder block.
    
    Consists of:
    1. Multi-head self-attention with pre-LayerNorm
    2. Feed-forward network with pre-LayerNorm
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        
        # Layer normalization (pre-norm architecture)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the encoder block.
        
        Args:
            x: Input of shape [batch_size, seq_len, d_model]
            attn_mask: Optional attention mask
            key_padding_mask: Optional padding mask
            need_weights: Whether to return attention weights (uses more memory)
            
        Returns:
            Tuple of (output, attention_weights)
            - output: Shape [batch_size, seq_len, d_model]
            - attention_weights: Shape [batch_size, n_heads, seq_len, seq_len] or None
        """
        # Self-attention with residual connection
        normed = self.norm1(x)
        attn_output, attn_weights = self.self_attn(
            normed, normed, normed,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        x = x + self.dropout(attn_output)
        
        # Feed-forward with residual connection
        normed = self.norm2(x)
        ff_output = self.feed_forward(normed)
        x = x + ff_output
        
        return x, attn_weights


class LIBSTransformer(nn.Module):
    """
    LIBS Foundation Model Transformer.
    
    A transformer encoder for learning representations from LIBS spectra.
    Supports masked intensity prediction for self-supervised pre-training.
    
    Args:
        n_bins: Number of wavelength bins
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_ff: Feed-forward dimension
        dropout: Dropout rate
        n_classes: Number of classes for downstream classification
        spectral_meta: ``embedding_type='spectral_patch'`` only: {wavelength [n_px] nm,
            window_nm, stride_nm, n_samples, input_scale, saturation_level,
            min_saturated_px, pe_scale, pe_wl_min, pe_wl_max[, n_segments]}
            (``models.spectral_patch_embedding.spectral_meta_from_config``)
    """
    
    def __init__(
        self,
        n_bins: int = 2048,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        n_classes: int = 5,
        embedding_type: str = "intensity",
        n_lines: int = 0,
        line_dict_meta: Optional[dict] = None,
        n_elements_vocab: int = 53,
        n_ion_states: int = 10,
        line_token_meta: Optional[dict] = None,
        n_mip_target_channels: int = 2,
        mip_loss_type: str = "mse",
        num_intensity_bins: int = 256,
        num_fwhm_bins: int = 100,
        spectral_meta: Optional[dict] = None,
    ):
        super().__init__()
        
        self.n_bins = n_bins
        self.n_lines = n_lines
        self.d_model = d_model
        self.n_layers = n_layers
        self.embedding_type = embedding_type
        self.n_mip_target_channels = int(n_mip_target_channels)
        self.mip_loss_type = mip_loss_type
        self.num_intensity_bins = int(num_intensity_bins)
        self.num_fwhm_bins = int(num_fwhm_bins)
        
        if embedding_type == "line_token":
            self.embedding = LineTokenEmbedding(
                d_model=d_model,
                n_elements=n_elements_vocab,
                n_ion_states=n_ion_states,
                dropout=dropout,
                dict_meta=line_dict_meta,
            )
            if line_dict_meta is not None:
                self.n_lines = int(line_dict_meta.get("n_lines", n_lines))
        elif embedding_type == "line_token_linear":
            if line_token_meta is None:
                raise ValueError(
                    "embedding_type='line_token_linear' requires line_token_meta "
                    "(use prepare_line_tokens_assets)"
                )
            self.embedding = LinearLineTokenEmbedding(
                n_features=line_token_meta["n_features"],
                d_model=d_model,
                feature_mean=line_token_meta["feature_mean"],
                feature_std=line_token_meta["feature_std"],
                central_wavelength=line_token_meta["central_wavelength"],
                dropout=dropout,
            )
            self.n_lines = int(line_token_meta["n_lines"])
            self.n_features = int(line_token_meta["n_features"])
        elif embedding_type == "spectral_patch":
            if spectral_meta is None or spectral_meta.get("wavelength") is None:
                raise ValueError(
                    "embedding_type='spectral_patch' requires spectral_meta with the "
                    "wavelength axis (spectral_meta_from_config)"
                )
            meta = {k: v for k, v in spectral_meta.items() if k != "wavelength"}
            self.embedding = SpectralPatchEmbedding(
                d_model=d_model,
                wavelength=spectral_meta["wavelength"],
                dropout=dropout,
                **meta,
            )
            self.n_bins = self.embedding.n_px
            self.n_tokens = self.embedding.n_tokens
        else:
            self.embedding = SpectralEmbedding(
                d_model=d_model,
                n_bins=n_bins,
                dropout=dropout,
            )
        
        # Transformer encoder blocks
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        
        # Final layer normalization
        self.final_norm = nn.LayerNorm(d_model)
        
        # Masked prediction head (regression or classification)
        if embedding_type == "spectral_patch":
            if mip_loss_type != "mse":
                raise ValueError("spectral_patch supports pretrain.loss 'mse' only")
            # reconstructs the transformed samples of each masked window
            self.mip_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
                nn.Linear(d_model, self.embedding.n_samples),
            )
        elif mip_loss_type == "classification":
            if embedding_type in ("line_token", "line_token_linear"):
                self.mip_head = MaskedLineFeatureHead(
                    d_model=d_model,
                    num_intensity_bins=num_intensity_bins,
                    num_fwhm_bins=num_fwhm_bins,
                )
            else:
                self.mip_head = MaskedBinIntensityHead(
                    d_model=d_model,
                    num_intensity_bins=num_intensity_bins,
                )
        elif embedding_type in ("line_token", "line_token_linear"):
            out_dim = 2 if embedding_type == "line_token" else self.n_mip_target_channels
            self.mip_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
                nn.Linear(d_model, out_dim),
            )
        else:
            self.mip_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 1),
            )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: Union[torch.Tensor, Dict[str, torch.Tensor]],
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the transformer.
        
        Args:
            x: [B, n_bins] spectrum (intensity / spectral_patch mode; a dict with
               'spectrum' also works for spectral_patch) or dict with 'line_features' [B, L, 6]
            mask: Optional boolean mask: [B, n_bins] (intensity mode) or [B, T] masked
               windows (spectral_patch mode)
            key_padding_mask: Optional [B, seq_len] True = ignore (line mode; from embedding)
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary with cls_embedding, sequence_output, mip_predictions, ...
        """
        if self.embedding_type == "line_token":
            if isinstance(x, dict):
                line_features = x["line_features"]
            else:
                line_features = x
            hidden, kpm = self.embedding(line_features)
            if key_padding_mask is None:
                key_padding_mask = kpm
        elif self.embedding_type == "line_token_linear":
            if isinstance(x, dict):
                tokens = x["tokens"]
                fit_valid = x.get("fit_valid")
            else:
                tokens, fit_valid = x, None
            hidden, kpm = self.embedding(tokens, fit_valid=fit_valid)
            if key_padding_mask is None:
                key_padding_mask = kpm
        elif self.embedding_type == "spectral_patch":
            spectrum = x["spectrum"] if isinstance(x, dict) else x
            hidden = self.embedding(spectrum, token_mask=mask)
        else:
            hidden = self.embedding(x, mask=mask)
        
        attention_weights = []
        for block in self.encoder_blocks:
            hidden, attn = block(
                hidden,
                key_padding_mask=key_padding_mask,
                need_weights=return_attention,
            )
            if return_attention and attn is not None:
                attention_weights.append(attn)
        
        hidden = self.final_norm(hidden)
        cls_embedding = hidden[:, 0, :]
        sequence_embeddings = hidden[:, 1:, :]
        
        result = {
            'cls_embedding': cls_embedding,
            'sequence_output': hidden,
            'sequence_embeddings': sequence_embeddings,
        }

        if self.mip_loss_type == "classification":
            if self.embedding_type in ("line_token", "line_token_linear"):
                intensity_logits, fwhm_logits = self.mip_head(sequence_embeddings)
                result['intensity_logits'] = intensity_logits
                result['fwhm_logits'] = fwhm_logits
            else:
                result['intensity_logits'] = self.mip_head(sequence_embeddings)
        else:
            mip_out = self.mip_head(sequence_embeddings)
            if self.embedding_type in ("line_token", "line_token_linear"):
                result['mip_predictions'] = mip_out  # [B, L, n_target_channels]
            elif self.embedding_type == "spectral_patch":
                result['mip_predictions'] = mip_out  # [B, T, n_samples]
            else:
                result['mip_predictions'] = mip_out.squeeze(-1)  # [B, n_bins]
        if key_padding_mask is not None:
            result['key_padding_mask'] = key_padding_mask
        if return_attention:
            result['attention_weights'] = attention_weights
        return result
    
    def get_embeddings(
        self,
        x: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """CLS embeddings for downstream tasks."""
        output = self.forward(x, mask=None, return_attention=False)
        return output['cls_embedding']
    
    def compute_mip_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """MSE over masked positions (bins or line channels)."""
        if predictions.dim() == 3:
            # Line-token: predictions/targets [B, L, C], mask [B, L] or [B, L, C]
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1).expand_as(predictions)
            masked_predictions = predictions[mask]
            masked_targets = targets[mask]
        else:
            masked_predictions = predictions[mask]
            masked_targets = targets[mask]
        
        if masked_predictions.numel() == 0:
            return torch.tensor(0.0, device=predictions.device)
        return nn.functional.mse_loss(masked_predictions, masked_targets)
    
    @property
    def num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze_encoder(self):
        """Freeze encoder weights for fine-tuning."""
        for param in self.embedding.parameters():
            param.requires_grad = False
        for block in self.encoder_blocks:
            for param in block.parameters():
                param.requires_grad = False
        for param in self.final_norm.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze encoder weights."""
        for param in self.parameters():
            param.requires_grad = True


if __name__ == "__main__":
    # Test the model
    batch_size = 4
    n_bins = 2048
    d_model = 256
    
    model = LIBSTransformer(
        n_bins=n_bins,
        d_model=d_model,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        dropout=0.1,
    )
    
    print(f"Model Parameters: {model.num_parameters:,}")
    
    # Test forward pass without mask
    x = torch.rand(batch_size, n_bins)
    output = model(x)
    print(f"\nForward pass (no mask):")
    print(f"  CLS embedding shape: {output['cls_embedding'].shape}")
    print(f"  Sequence output shape: {output['sequence_output'].shape}")
    print(f"  MIP predictions shape: {output['mip_predictions'].shape}")
    
    # Test forward pass with mask
    mask = torch.rand(batch_size, n_bins) < 0.15
    output = model(x, mask=mask, return_attention=True)
    print(f"\nForward pass (with mask):")
    print(f"  CLS embedding shape: {output['cls_embedding'].shape}")
    print(f"  MIP predictions shape: {output['mip_predictions'].shape}")
    print(f"  Number of attention layers: {len(output['attention_weights'])}")
    print(f"  Attention shape per layer: {output['attention_weights'][0].shape}")
    
    # Test MIP loss computation
    targets = torch.rand(batch_size, n_bins)
    loss = model.compute_mip_loss(output['mip_predictions'], targets, mask)
    print(f"\nMIP Loss: {loss.item():.4f}")
    
    # Test embeddings extraction
    embeddings = model.get_embeddings(x)
    print(f"\nEmbeddings shape: {embeddings.shape}")
