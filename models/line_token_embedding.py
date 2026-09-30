"""
Line-as-token embeddings.

Two flavours are provided:

- ``LineTokenEmbedding``         — runtime concatenation of static quantum
  parameters + element/ion learnable embeddings + 5 Voigt-fit channels; the
  embedding owns buffers for every line.

- ``LinearLineTokenEmbedding``   — consumes the pre-baked ``line_tokens_*.h5``
  cache (data/line_tokenization.py). Each line is already a 14-feature
  float32 vector; the embedding is just z-score normalization + ``nn.Linear``,
  plus wavelength PE and CLS. This is the recommended path now that
  tokenization is a separate, reusable preprocessing step.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from data.line_features import FEAT_VALID


class DynamicWavelengthEncoding(nn.Module):
    """Sinusoidal PE from physical wavelength (nm), not bin index.

    The wavelength is normalised to [0, 1] over ``wl_min``..``wl_max`` and
    multiplied by ``scale``, so the finest angular frequency is
    ``scale / (wl_max - wl_min)`` rad/nm.
    """

    def __init__(
        self,
        d_model: int,
        wl_min: float,
        wl_max: float,
        dropout: float = 0.1,
        scale: float = 1000.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.wl_min = wl_min
        self.wl_max = max(wl_max, wl_min + 1e-6)
        self.scale = float(scale)
        self.dropout = nn.Dropout(p=dropout)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, x: torch.Tensor, wavelengths_nm: torch.Tensor) -> torch.Tensor:
        wl_norm = (wavelengths_nm - self.wl_min) / (self.wl_max - self.wl_min)
        wl_norm = wl_norm.clamp(0.0, 1.0)
        pe = torch.zeros(x.size(0), x.size(1), self.d_model, device=x.device, dtype=x.dtype)
        pe[:, :, 0::2] = torch.sin(wl_norm.unsqueeze(-1) * self.scale * self.div_term)
        pe[:, :, 1::2] = torch.cos(wl_norm.unsqueeze(-1) * self.scale * self.div_term)
        return self.dropout(x + pe)


class LineTokenEmbedding(nn.Module):
    """
    Embed each spectral line as one transformer token.

    Static (dictionary): wavelength, Ei, Ek, log(gi/gk/Ak/I_theory), element & ion embeddings.
    Dynamic (per spectrum): normalized Voigt fit features (5 channels).
    """

    N_QUANTUM_SCALARS = 7

    def __init__(
        self,
        d_model: int = 256,
        n_elements: int = 53,
        n_ion_states: int = 10,
        element_emb_dim: int = 16,
        ion_emb_dim: int = 4,
        dropout: float = 0.1,
        dict_meta: Optional[dict] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_lines = 0

        if dict_meta is not None:
            self._register_dict_buffers(dict_meta)
        else:
            self.register_buffer("quant_static", torch.zeros(0, self.N_QUANTUM_SCALARS))
            self.register_buffer("wl_min", torch.tensor(200.0))
            self.register_buffer("wl_max", torch.tensor(900.0))
            self.register_buffer("quant_mean", torch.zeros(self.N_QUANTUM_SCALARS))
            self.register_buffer("quant_std", torch.ones(self.N_QUANTUM_SCALARS))
            self.register_buffer("element_id", torch.zeros(0, dtype=torch.long))
            self.register_buffer("ion_state_id", torch.zeros(0, dtype=torch.long))
            self.register_buffer("central_wavelength", torch.zeros(0))

        self.register_buffer("fit_mean", torch.zeros(5))
        self.register_buffer("fit_std", torch.ones(5))

        self.element_embedding = nn.Embedding(n_elements, element_emb_dim)
        self.ion_embedding = nn.Embedding(n_ion_states, ion_emb_dim)

        in_dim = self.N_QUANTUM_SCALARS + element_emb_dim + ion_emb_dim + 5
        self.projection = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_encoding = DynamicWavelengthEncoding(
            d_model=d_model,
            wl_min=float(self.wl_min),
            wl_max=float(self.wl_max),
            dropout=dropout,
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def _register_dict_buffers(self, meta: dict) -> None:
        wl = meta["central_wavelength"].astype(np.float64)
        gi = np.maximum(meta["gi"], 1e-30)
        gk = np.maximum(meta["gk"], 1e-30)
        Ak = np.maximum(meta["Ak"], 1e-30)
        Ith = np.maximum(meta["theoretical_intensity"], 1e-30)
        quant = np.stack([
            wl,
            meta["Ei"],
            meta["Ek"],
            np.log10(gi),
            np.log10(gk),
            np.log10(Ak),
            np.log10(Ith),
        ], axis=1).astype(np.float32)
        mean = quant.mean(axis=0)
        std = quant.std(axis=0)
        std[std < 1e-8] = 1.0

        self.n_lines = int(wl.size)
        self.register_buffer("quant_static", torch.from_numpy(quant))
        self.register_buffer("wl_min", torch.tensor(float(wl.min())))
        self.register_buffer("wl_max", torch.tensor(float(wl.max())))
        self.register_buffer("quant_mean", torch.from_numpy(mean))
        self.register_buffer("quant_std", torch.from_numpy(std))
        self.register_buffer("central_wavelength", torch.from_numpy(wl.astype(np.float32)))
        self.register_buffer("element_id", torch.from_numpy(meta["element_id"].astype(np.int64)))
        self.register_buffer("ion_state_id", torch.from_numpy(meta["ion_state_id"].astype(np.int64)))

        if "fit_mean" in meta:
            self.register_buffer("fit_mean", torch.from_numpy(np.asarray(meta["fit_mean"], dtype=np.float32)))
            fit_std = np.asarray(meta["fit_std"], dtype=np.float32)
            fit_std[fit_std < 1e-8] = 1.0
            self.register_buffer("fit_std", torch.from_numpy(fit_std))

    def set_fit_normalization(self, fit_mean: np.ndarray, fit_std: np.ndarray) -> None:
        self.fit_mean.copy_(torch.from_numpy(fit_mean.astype(np.float32)))
        std = fit_std.astype(np.float32).copy()
        std[std < 1e-8] = 1.0
        self.fit_std.copy_(torch.from_numpy(std))

    def forward(
        self,
        line_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            line_features: [B, n_lines, 6]

        Returns:
            tokens: [B, n_lines + 1, d_model]
            key_padding_mask: [B, n_lines + 1] True = ignore
        """
        B, L, _ = line_features.shape
        device = line_features.device

        fit = line_features[:, :, :5]
        valid = line_features[:, :, FEAT_VALID] > 0.5
        fit_norm = (fit - self.fit_mean) / self.fit_std

        quant_norm = (self.quant_static.unsqueeze(0).expand(B, -1, -1) - self.quant_mean) / self.quant_std
        elem_emb = self.element_embedding(self.element_id.to(device)).unsqueeze(0).expand(B, -1, -1)
        ion_emb = self.ion_embedding(self.ion_state_id.to(device)).unsqueeze(0).expand(B, -1, -1)

        token_in = torch.cat([quant_norm, elem_emb, ion_emb, fit_norm], dim=-1)
        x = self.projection(token_in)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        wl = self.central_wavelength.to(device).unsqueeze(0).expand(B, -1)
        wl_cls = torch.zeros(B, 1, device=device, dtype=wl.dtype)
        x = self.pos_encoding(x, torch.cat([wl_cls, wl], dim=1))
        x = self.layer_norm(x)

        valid_cls = torch.ones(B, 1, dtype=torch.bool, device=device)
        pad_mask = torch.cat([valid_cls, ~valid], dim=1)
        return x, pad_mask


class LinearLineTokenEmbedding(nn.Module):
    """
    Simplest possible line-token embedding for the pre-tokenized cache.

    Inputs (from a ``line_tokens_<hash>.h5`` reader):
        tokens     [B, n_lines, n_features]   raw float32 features
        fit_valid  [B, n_lines]               uint8/bool, 1 = valid Voigt fit

    Pipeline:
        z-score normalize using stored mean/std →
        nn.Linear(n_features, d_model) →
        prepend learnable CLS →
        wavelength positional encoding (sinusoidal on physical λ in nm) →
        LayerNorm

    Returns:
        x          [B, n_lines + 1, d_model]
        kpm        [B, n_lines + 1] True = ignore (CLS always kept)
    """

    def __init__(
        self,
        n_features: int,
        d_model: int,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        central_wavelength: np.ndarray,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = int(n_features)
        self.d_model = int(d_model)
        self.n_lines = int(central_wavelength.shape[0])

        mean = np.asarray(feature_mean, dtype=np.float32).reshape(-1)
        std = np.asarray(feature_std, dtype=np.float32).reshape(-1).copy()
        std[std < 1e-8] = 1.0
        if mean.size != self.n_features or std.size != self.n_features:
            raise ValueError(
                f"feature_mean/std must have length {self.n_features} "
                f"(got {mean.size}/{std.size})"
            )
        wl = np.asarray(central_wavelength, dtype=np.float32).reshape(-1)

        self.register_buffer("feature_mean", torch.from_numpy(mean))
        self.register_buffer("feature_std", torch.from_numpy(std))
        self.register_buffer("central_wavelength", torch.from_numpy(wl))

        self.projection = nn.Linear(self.n_features, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_encoding = DynamicWavelengthEncoding(
            d_model=d_model,
            wl_min=float(wl.min()),
            wl_max=float(wl.max()),
            dropout=dropout,
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tokens: torch.Tensor,
        fit_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if tokens.dim() != 3 or tokens.size(-1) != self.n_features:
            raise ValueError(
                f"tokens must be [B, n_lines, {self.n_features}], got {tuple(tokens.shape)}"
            )
        B, L, _ = tokens.shape
        device = tokens.device

        normed = (tokens - self.feature_mean) / self.feature_std
        x = self.projection(normed)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        wl = self.central_wavelength.to(device).unsqueeze(0).expand(B, -1)
        wl_cls = torch.zeros(B, 1, device=device, dtype=wl.dtype)
        x = self.pos_encoding(x, torch.cat([wl_cls, wl], dim=1))
        x = self.layer_norm(x)

        if fit_valid is None:
            pad_mask = torch.zeros(B, L + 1, dtype=torch.bool, device=device)
        else:
            valid = fit_valid.to(device=device, dtype=torch.bool)
            cls_valid = torch.ones(B, 1, dtype=torch.bool, device=device)
            pad_mask = torch.cat([cls_valid, ~valid], dim=1)
        return x, pad_mask
