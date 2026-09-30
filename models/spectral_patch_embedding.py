"""
Raw-spectrum window embedding (``embedding_type: spectral_patch``).

The unit-normalised spectrum is read through overlapping windows of fixed
physical width, so no line dictionary, Voigt fit or token cache is involved:

    spectrum [B, n_px] (native pixel order, any number of detector channels)
      -> stitched monotonic axis (overlapping channels: the earlier channel wins)
      -> T windows of ``window_nm`` every ``stride_nm``, each resampled to
         ``n_samples`` points by linear interpolation            [B, T, n_samples]
      -> per sample: asinh(x / input_scale) and a saturation flag   [B, T, 2*n_samples]
      -> one shared nn.Linear(2*n_samples, d_model)
         + detector-segment embedding + sinusoidal PE of the window centre (nm)
      -> CLS prepended (no PE) -> LayerNorm                         [B, T + 1, d_model]

Window centres sit on the global grid ``k * stride_nm`` and the PE is taken
over a fixed reference range (``pe_wl_min``..``pe_wl_max``), so a token means
the same wavelength interval on every instrument that covers it; only the
detector-segment table depends on the axis.  Index tables are derived from
the wavelength axis at construction and are not stored in checkpoints
(``persistent=False``); consumers rebuild the embedding from the axis recorded
in ``run_info.yaml`` (``spectral_patch.axis``).

Masked-window pretraining (``LIBSPretrainModule``): ``sample_token_mask``
draws span masks on the GPU; ``forward(x, token_mask)`` zeroes every stitched
pixel read by a masked window before windowing (overlapping neighbours would
otherwise leak it) and swaps the masked token embeddings for ``mask_token``;
``window_targets`` gives the clean asinh windows to reconstruct.

Units: wavelengths in nm; intensities unit-normalised (``libs_pipeline.unit_norm``).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .line_token_embedding import DynamicWavelengthEncoding

DEFAULT_PATCH_CFG: dict[str, float] = {
    "window_nm": 1.0,  # window width (nm)
    "stride_nm": 0.5,  # centre spacing (nm); windows overlap by window - stride
    "n_samples": 48,  # interpolation points per window
    "input_scale": 1.0e-2,  # asinh(x / input_scale): ~linear below, ~log above
    "saturation_level": 0.999,  # unit-normalised level counted as clipped
    "min_saturated_px": 3,  # a spectrum is saturated with >= this many clipped pixels
    "pe_scale": 3000.0,  # DynamicWavelengthEncoding frequency scale
    "pe_wl_min": 180.0,  # fixed PE reference range (nm), instrument independent
    "pe_wl_max": 900.0,
}


def detector_segment_bounds(wavelength: np.ndarray, spacing_jump: float = 0.2) -> list[int]:
    """Pixel bounds ``[0, b1, ..., n_px]`` of the detector segments: a new segment
    starts where the axis stops increasing (channel seam) or where the pixel
    spacing changes by more than ``spacing_jump`` (relative) between neighbours."""
    wl = np.asarray(wavelength, dtype=np.float64).reshape(-1)
    d = np.diff(wl)
    jump = np.abs(np.diff(d)) > spacing_jump * np.abs(d[:-1])
    cand = sorted(
        set((np.nonzero(d <= 0)[0] + 1).tolist()) | set((np.nonzero(jump)[0] + 1).tolist())
    )
    bounds: list[int] = []
    for c in cand:  # a seam shows up as a pair (k, k+1): keep the later index
        if bounds and c - bounds[-1] <= 1:
            bounds[-1] = c
        else:
            bounds.append(c)
    return [0, *[b for b in bounds if 0 < b < wl.size], int(wl.size)]


def stitched_mask(wavelength: np.ndarray) -> np.ndarray:
    """Pixels kept on the monotonic (stitched) axis: pixel i is kept when its
    wavelength exceeds every kept pixel before it, i.e. in overlapping channel
    regions the earlier channel wins."""
    wl = np.asarray(wavelength, dtype=np.float64).reshape(-1)
    running_max = np.maximum.accumulate(np.concatenate([[-np.inf], wl[:-1]]))
    return wl > running_max


def axis_signature(wavelength: np.ndarray) -> dict[str, Any]:
    """Compact identity of a wavelength axis for run_info.yaml."""
    wl = np.asarray(wavelength, dtype=np.float64).reshape(-1)
    return {
        "n_px": int(wl.size),
        "first_nm": round(float(wl[0]), 4),
        "last_nm": round(float(wl[-1]), 4),
        "n_segments": len(detector_segment_bounds(wl)) - 1,
        "md5": hashlib.md5(np.round(wl, 4).tobytes()).hexdigest()[:12],
    }


def patch_config(model_cfg: Mapping[str, Any] | None) -> dict[str, float]:
    """``model.patch`` block of a model config merged over the defaults."""
    cfg = dict(DEFAULT_PATCH_CFG)
    cfg.update(dict((model_cfg or {}).get("patch") or {}))
    return cfg


def spectral_meta_from_config(
    model_cfg: Mapping[str, Any] | None, wavelength: np.ndarray, **overrides: Any
) -> dict[str, Any]:
    """``spectral_meta`` for ``LIBSTransformer(embedding_type='spectral_patch')``."""
    return {**patch_config(model_cfg), **overrides, "wavelength": np.asarray(wavelength)}


def wavelength_from_run_info(run_info: Mapping[str, Any], root: str | Path = ".") -> np.ndarray:
    """Training wavelength axis of a spectral_patch run: ``spectral_patch.axis.source``
    (the data config's ``paths.wavelength_json``), checked against the recorded md5."""
    from data.libs_pipeline import load_wavelength  # lazy: models stay importable alone

    axis = dict((run_info.get("spectral_patch") or {}).get("axis") or {})
    source = axis.get("source")
    if not source:
        raise ValueError("run_info.yaml has no spectral_patch.axis.source")
    path = Path(source)
    path = path if path.is_absolute() else Path(root) / path
    wl = np.asarray(load_wavelength(str(path)), dtype=np.float64)
    md5 = axis_signature(wl)["md5"]
    if axis.get("md5") and md5 != axis["md5"]:
        raise ValueError(
            f"wavelength axis {path} (md5 {md5}) differs from training ({axis['md5']})"
        )
    return wl


def spectral_patch_run_info(
    spectral_meta: Mapping[str, Any] | None, model: nn.Module, libs_data_config: str | None
) -> dict[str, Any] | None:
    """``spectral_patch`` block of run_info.yaml: window settings, token count and the
    training wavelength axis (source file + signature), from which consumers rebuild
    the embedding (:func:`wavelength_from_run_info`). None for other embedding types."""
    if spectral_meta is None:
        return None
    import yaml  # lazy

    source = None
    if libs_data_config:
        with open(libs_data_config) as f:
            source = ((yaml.safe_load(f) or {}).get("paths") or {}).get("wavelength_json")
    emb = model.embedding
    settings = {
        k: float(v) for k, v in spectral_meta.items()
        if k != "wavelength" and isinstance(v, (int, float))
    }  # fmt: skip
    return {
        **settings,
        "n_samples": int(emb.n_samples),
        "min_saturated_px": int(emb.min_saturated_px),
        "n_segments": int(emb.n_segments),
        "n_tokens": int(emb.n_tokens),
        "axis": {**axis_signature(spectral_meta["wavelength"]), "source": source},
    }


class SpectralPatchEmbedding(nn.Module):
    """Overlapping fixed-width windows of the raw spectrum as transformer tokens.

    Args:
        d_model:           model dimension
        wavelength:        spectrometer axis [n_px] in nm (may be non-monotonic)
        window_nm, stride_nm, n_samples, input_scale, saturation_level,
        min_saturated_px, pe_scale, pe_wl_min, pe_wl_max: see DEFAULT_PATCH_CFG
        n_segments:        size of the detector-segment table (default: segments
                           of ``wavelength``; pass the training value when the
                           embedding is rebuilt on another axis)
        dropout:           dropout of the positional encoding

    Forward:
        x           [B, n_px] unit-normalised spectra
        token_mask  [B, T] bool, True = masked window (pretraining only)
        returns     [B, T + 1, d_model]
    """

    def __init__(
        self,
        d_model: int,
        wavelength: np.ndarray,
        window_nm: float = 1.0,
        stride_nm: float = 0.5,
        n_samples: int = 48,
        input_scale: float = 1.0e-2,
        saturation_level: float = 0.999,
        min_saturated_px: int = 3,
        pe_scale: float = 3000.0,
        pe_wl_min: float = 180.0,
        pe_wl_max: float = 900.0,
        n_segments: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        wl = np.asarray(wavelength, dtype=np.float64).reshape(-1)
        if wl.size < 4:
            raise ValueError("spectral_patch needs a wavelength axis with at least 4 pixels")
        self.d_model = int(d_model)
        self.n_px = int(wl.size)
        self.window_nm = float(window_nm)
        self.stride_nm = float(stride_nm)
        self.n_samples = int(n_samples)
        self.input_scale = float(input_scale)
        self.saturation_level = float(saturation_level)
        self.min_saturated_px = int(min_saturated_px)
        if self.window_nm <= 0 or self.stride_nm <= 0 or self.n_samples < 2:
            raise ValueError("window_nm and stride_nm must be > 0 and n_samples >= 2")

        bounds = detector_segment_bounds(wl)
        seg_of_px = np.zeros(wl.size, dtype=np.int64)
        for s, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
            seg_of_px[a:b] = s
        self.n_segments = int(n_segments) if n_segments else len(bounds) - 1

        keep = stitched_mask(wl)
        px_s = np.nonzero(keep)[0]
        wl_s = wl[keep]
        n_keep = wl_s.size
        half = 0.5 * self.window_nm
        k0 = int(np.ceil((wl_s[0] + half) / self.stride_nm - 1e-9))
        k1 = int(np.floor((wl_s[-1] - half) / self.stride_nm + 1e-9))
        if k1 < k0:
            raise ValueError(f"axis {wl_s[0]:.2f}-{wl_s[-1]:.2f} nm is narrower than one window")
        centres = np.arange(k0, k1 + 1, dtype=np.float64) * self.stride_nm
        if centres[0] < pe_wl_min or centres[-1] > pe_wl_max:
            # DynamicWavelengthEncoding clamps outside its range: windows there would
            # share one positional encoding
            raise ValueError(
                f"window centres {centres[0]:.1f}-{centres[-1]:.1f} nm fall outside the PE "
                f"reference range {pe_wl_min:g}-{pe_wl_max:g} nm; widen model.patch.pe_wl_min/max"
            )
        offsets = np.linspace(-half, half, self.n_samples)
        pos = centres[:, None] + offsets[None, :]  # [T, n_samples] nm
        j1 = np.clip(np.searchsorted(wl_s, pos, side="right"), 1, n_keep - 1)
        j0 = j1 - 1
        frac = np.clip((pos - wl_s[j0]) / (wl_s[j1] - wl_s[j0]), 0.0, 1.0)
        # stitched pixels read by each window, and for each stitched pixel the
        # contiguous range of windows that read it (centres are sorted)
        s_lo, s_hi = j0.min(axis=1), j1.max(axis=1)
        pix = np.arange(n_keep)
        t_first = np.searchsorted(s_hi, pix, side="left")
        t_last = np.searchsorted(s_lo, pix, side="right") - 1
        seg_tok = seg_of_px[px_s[np.clip(np.searchsorted(wl_s, centres), 0, n_keep - 1)]]
        self.n_tokens = int(centres.size)

        self.register_buffer("px_stitched", torch.from_numpy(px_s), persistent=False)
        self.register_buffer("idx0", torch.from_numpy(j0), persistent=False)
        self.register_buffer("idx1", torch.from_numpy(j1), persistent=False)
        self.register_buffer("frac", torch.from_numpy(frac.astype(np.float32)), persistent=False)
        self.register_buffer("t_first", torch.from_numpy(t_first), persistent=False)
        self.register_buffer("t_last", torch.from_numpy(t_last), persistent=False)
        self.register_buffer(
            "token_centre_nm", torch.from_numpy(centres.astype(np.float32)), persistent=False
        )
        self.register_buffer(
            "token_segment",
            torch.from_numpy(np.minimum(seg_tok, self.n_segments - 1)),
            persistent=False,
        )

        self.patch_proj = nn.Linear(2 * self.n_samples, self.d_model)
        self.segment_embedding = nn.Embedding(self.n_segments, self.d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
        self.pos_encoding = DynamicWavelengthEncoding(
            d_model=self.d_model,
            wl_min=float(pe_wl_min),
            wl_max=float(pe_wl_max),
            dropout=dropout,
            scale=float(pe_scale),
        )
        self.layer_norm = nn.LayerNorm(self.d_model)

    # ── windows ──
    def _stitched(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.size(1) != self.n_px:
            raise ValueError(f"spectra must be [B, {self.n_px}], got {tuple(x.shape)}")
        return x.float().index_select(1, self.px_stitched)

    def _windows(self, xs: torch.Tensor) -> torch.Tensor:
        """Linear interpolation of stitched spectra [B, n_keep] -> [B, T, n_samples]."""
        return xs[:, self.idx0] * (1.0 - self.frac) + xs[:, self.idx1] * self.frac

    def transform(self, w: torch.Tensor) -> torch.Tensor:
        return torch.asinh(w / self.input_scale)

    def saturation_flags(self, xs: torch.Tensor) -> torch.Tensor:
        """1.0 at clipped pixels of spectra that have a clipped plateau, else 0.0."""
        at_level = xs >= self.saturation_level
        saturated = at_level.sum(dim=1, keepdim=True) >= self.min_saturated_px
        return (at_level & saturated).float()

    def window_targets(self, x: torch.Tensor) -> torch.Tensor:
        """Clean transformed windows [B, T, n_samples] (masked-window reconstruction target)."""
        return self.transform(self._windows(self._stitched(x)))

    # ── masking ──
    def sample_token_mask(
        self,
        batch_size: int,
        mask_ratio: float,
        span_tokens: int = 1,
        device: torch.device | str | None = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Span masking [B, T]: each window is masked with probability ``mask_ratio``
        (in expectation, including the first windows), in spans of ``span`` consecutive
        windows. Span starts are drawn over T + span - 1 positions (spans may begin
        before window 0) with p = 1 - (1 - mask_ratio)^(1/span), so that the union
        of overlapping spans covers ``mask_ratio`` of the windows; at least one
        window per row."""
        T = self.n_tokens
        span = max(1, int(span_tokens))
        ratio = min(max(float(mask_ratio), 0.0), 1.0)
        p = 1.0 - (1.0 - ratio) ** (1.0 / span)
        u = torch.rand(batch_size, T + span - 1, device=device, generator=generator)
        starts = (u < p).float().unsqueeze(1)
        if span > 1:  # start s masks windows s - span + 1 .. s (shifted by span - 1)
            starts = F.max_pool1d(starts, kernel_size=span, stride=1)
        mask = starts.squeeze(1) > 0  # [B, T]
        empty = ~mask.any(dim=1)
        if empty.any():
            fallback = torch.argmin(u[:, span - 1 :], dim=1)  # the smallest draw of the row
            mask[empty, fallback[empty]] = True
        return mask

    def pixel_mask(self, token_mask: torch.Tensor) -> torch.Tensor:
        """Stitched pixels [B, n_keep] read by at least one masked window."""
        cm = F.pad(
            token_mask.long().cumsum(dim=1), (1, 0)
        )  # [B, T + 1], cm[:, t] = masked in [0, t)
        hi = (self.t_last + 1).clamp(0, self.n_tokens)
        lo = self.t_first.clamp(0, self.n_tokens)
        return (cm[:, hi] - cm[:, lo]) > 0

    # ── forward ──
    def forward(self, x: torch.Tensor, token_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        xs = self._stitched(x)
        if token_mask is not None:
            token_mask = token_mask.to(device=xs.device, dtype=torch.bool)
            xs = xs.masked_fill(self.pixel_mask(token_mask), 0.0)
        # flags from the visible pixels only: the per-spectrum plateau count must not
        # see pixels hidden by masked windows
        flags = self.saturation_flags(xs)
        feats = torch.cat([self.transform(self._windows(xs)), self._windows(flags)], dim=-1)
        h = self.patch_proj(feats) + self.segment_embedding(self.token_segment)
        B, T = h.shape[0], h.shape[1]
        if token_mask is not None:
            h = torch.where(
                token_mask.unsqueeze(-1), self.mask_token.to(h.dtype).expand(B, T, -1), h
            )
        h = self.pos_encoding(h, self.token_centre_nm.unsqueeze(0).expand(B, -1))
        h = torch.cat([self.cls_token.to(h.dtype).expand(B, -1, -1), h], dim=1)
        return self.layer_norm(h)

    def extra_repr(self) -> str:
        return (
            f"n_px={self.n_px}, n_tokens={self.n_tokens}, window_nm={self.window_nm}, "
            f"stride_nm={self.stride_nm}, n_samples={self.n_samples}, n_segments={self.n_segments}"
        )
