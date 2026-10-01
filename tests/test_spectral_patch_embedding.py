"""Tests for models/spectral_patch_embedding.py (embedding_type 'spectral_patch')
and its integration in LIBSTransformer, LIBSPretrainModule and LIBSFinetuneModule."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from models.libs_transformer import LIBSTransformer
from models.line_token_embedding import DynamicWavelengthEncoding
from models.spectral_patch_embedding import (
    SpectralPatchEmbedding,
    axis_signature,
    detector_segment_bounds,
    spectral_meta_from_config,
    spectral_patch_run_info,
    stitched_mask,
)
from training.finetune import LIBSFinetuneModule
from training.pretrain import LIBSPretrainModule


def avantes_like_axis() -> np.ndarray:
    """Three channels with overlapping seams (non-monotonic) and a spacing change
    inside the second channel, like the 2022 Avantes set-up (scaled down)."""
    ch1 = np.arange(188.7, 260.01, 0.05)
    ch2a = np.arange(259.98, 370.0, 0.06)  # starts below the end of ch1
    ch2b = np.arange(370.0, 460.0, 0.045)  # spacing jump, monotonic
    ch3 = np.arange(459.97, 689.1, 0.25)  # starts below the end of ch2
    return np.concatenate([ch1, ch2a, ch2b, ch3])


WL = avantes_like_axis()
CFG = {"window_nm": 1.0, "stride_nm": 0.5, "n_samples": 24, "input_scale": 1e-2, "pe_scale": 3000.0}


def make_embedding(**kw) -> SpectralPatchEmbedding:
    return SpectralPatchEmbedding(d_model=16, wavelength=WL, dropout=0.0, **{**CFG, **kw})


def test_segments_and_stitching():
    bounds = detector_segment_bounds(WL)
    assert len(bounds) - 1 == 4  # two seams + one spacing jump
    keep = stitched_mask(WL)
    assert np.all(np.diff(WL[keep]) > 0)  # strictly increasing
    assert keep.sum() > WL.size - 10  # only the few overlapping seam pixels are dropped
    sig = axis_signature(WL)
    assert sig["n_px"] == WL.size and sig["n_segments"] == 4


def test_window_grid_and_resampling_match_numpy():
    emb = make_embedding()
    centres = emb.token_centre_nm.numpy().astype(np.float64)
    assert np.allclose(centres / 0.5, np.round(centres / 0.5))  # global k * stride grid
    keep = stitched_mask(WL)
    assert centres[0] - 0.5 >= WL[keep][0] - 1e-6 and centres[-1] + 0.5 <= WL[keep][-1] + 1e-6
    rng = np.random.default_rng(0)
    x = rng.random((2, WL.size)).astype(np.float32)
    got = emb._windows(emb._stitched(torch.from_numpy(x))).numpy()
    offsets = np.linspace(-0.5, 0.5, CFG["n_samples"])
    for t in (0, emb.n_tokens // 2, emb.n_tokens - 1):
        ref = np.interp(centres[t] + offsets, WL[keep], x[1, keep])
        assert np.allclose(got[1, t], ref, atol=1e-5)


def test_masking_zeroes_every_pixel_of_masked_windows():
    emb = make_embedding()
    x = torch.rand(3, WL.size) + 0.5  # strictly positive, so zeros can only come from masking
    tm = emb.sample_token_mask(3, 0.3, span_tokens=3, generator=torch.Generator().manual_seed(1))
    assert tm.any(dim=1).all()
    xs = emb._stitched(x)
    hidden = emb.pixel_mask(tm)
    masked_windows = emb._windows(xs.masked_fill(hidden, 0.0))
    # masked windows are empty (nothing leaks through overlapping neighbours) ...
    assert masked_windows[tm].abs().max() == 0
    # ... and exactly the stitched pixel span read by the masked windows is removed
    lo = torch.minimum(emb.idx0.min(dim=1).values, emb.idx1.min(dim=1).values)
    hi = torch.maximum(emb.idx0.max(dim=1).values, emb.idx1.max(dim=1).values)
    for b in range(3):
        span = torch.zeros(hidden.shape[1], dtype=torch.bool)
        for t in torch.nonzero(tm[b]).flatten().tolist():
            span[lo[t] : hi[t] + 1] = True
        assert torch.equal(hidden[b], span)


@pytest.mark.parametrize("ratio,span", [(0.3, 1), (0.3, 3), (0.6, 5)])
def test_mask_ratio_is_respected_everywhere(ratio, span):
    emb = make_embedding()
    g = torch.Generator().manual_seed(0)
    tm = emb.sample_token_mask(4000, ratio, span_tokens=span, generator=g).float()
    assert abs(tm.mean().item() - ratio) < 0.01
    assert abs(tm[:, 0].mean().item() - ratio) < 0.04  # the first window is not under-masked


def test_hidden_pixels_do_not_leak_through_saturation_flags():
    emb = make_embedding()
    tm = torch.zeros(1, emb.n_tokens, dtype=torch.bool)
    tm[0, emb.n_tokens // 2] = True
    hidden = emb.pixel_mask(tm)[0]
    stitched_px = emb.px_stitched[hidden]  # original pixel indices hidden by the mask
    visible_plateau = emb.px_stitched[~hidden][:2]  # 2 clipped pixels outside the masked window
    a = torch.full((1, WL.size), 0.2)
    a[0, visible_plateau] = 1.0
    b = a.clone()
    a[0, stitched_px[:2]] = 1.0  # hidden plateau pixels would complete the 3-pixel count
    b[0, stitched_px[:2]] = 0.3
    emb.eval()
    assert torch.equal(emb(a, token_mask=tm), emb(b, token_mask=tm))


def test_windows_outside_pe_range_are_rejected():
    with pytest.raises(ValueError, match="PE reference range"):
        make_embedding(pe_wl_min=200.0)


def test_saturation_flags_need_a_plateau():
    emb = make_embedding(min_saturated_px=3)
    x = torch.zeros(2, WL.size)
    x[0, 100] = 1.0  # unsaturated: a single maximum
    x[1, 100:104] = 1.0  # clipped plateau
    flags = emb.saturation_flags(emb._stitched(x))
    assert flags[0].sum() == 0 and flags[1].sum() >= 3


def _model(mip: str = "mse") -> LIBSTransformer:
    meta = spectral_meta_from_config({"patch": CFG}, WL)
    return LIBSTransformer(
        n_bins=WL.size, d_model=16, n_heads=2, n_layers=1, d_ff=32, dropout=0.0,
        embedding_type="spectral_patch", spectral_meta=meta, mip_loss_type=mip,
    )  # fmt: skip


def test_transformer_forward_shapes_and_finetune_pooling():
    m = _model()
    x = torch.rand(2, WL.size)
    out = m(x)
    T = m.embedding.n_tokens
    assert out["sequence_output"].shape == (2, T + 1, 16)
    assert out["mip_predictions"].shape == (2, T, CFG["n_samples"])
    assert "key_padding_mask" not in out
    ft = LIBSFinetuneModule(encoder=m, task="classification", n_classes=5, pool="cls_mean")
    o = ft({"spectrum": x})
    assert o["class_logits"].shape == (2, 5)
    with pytest.raises(ValueError, match="mse"):
        _model("classification")


def test_pretrain_step_runs_and_masks_differ_between_steps():
    m = _model()
    module = LIBSPretrainModule(model=m, mask_ratio=0.3, mask_span_tokens=3)
    batch = {"spectrum": torch.rand(4, WL.size)}
    loss, preds, targets = module._patch_step(batch)
    assert torch.isfinite(loss) and preds.shape == targets.shape and preds.numel() > 0
    loss.backward()
    assert m.embedding.patch_proj.weight.grad is not None
    g = torch.Generator().manual_seed(5)
    a = m.embedding.sample_token_mask(4, 0.3, 3, generator=g)
    b = m.embedding.sample_token_mask(4, 0.3, 3, generator=torch.Generator().manual_seed(5))
    assert torch.equal(a, b)  # seeded validation masks are reproducible


def test_embedding_rebuilds_on_another_axis_with_same_weights():
    m = _model()
    sd = {k: v for k, v in m.state_dict().items()}
    assert not any(k.startswith("embedding.idx") for k in sd)  # index tables are derived
    shifted = WL + 0.03  # e.g. another calibration of the same spectrometer
    meta = spectral_meta_from_config({"patch": CFG}, shifted, n_segments=m.embedding.n_segments)
    m2 = LIBSTransformer(
        n_bins=shifted.size, d_model=16, n_heads=2, n_layers=1, d_ff=32, dropout=0.0,
        embedding_type="spectral_patch", spectral_meta=meta,
    )  # fmt: skip
    m2.load_state_dict(sd, strict=True)  # same weights, tables rebuilt on the new axis
    assert m2(torch.rand(1, shifted.size))["cls_embedding"].shape == (1, 16)


def test_dynamic_wavelength_encoding_default_scale_unchanged():
    enc = DynamicWavelengthEncoding(d_model=8, wl_min=200.0, wl_max=400.0, dropout=0.0)
    x = torch.zeros(1, 3, 8)
    wl = torch.tensor([[200.0, 300.0, 400.0]])
    got = enc(x, wl)
    norm = (wl - 200.0) / 200.0
    div = torch.exp(torch.arange(0, 8, 2).float() * (-math.log(10000.0) / 8))
    assert torch.allclose(got[..., 0::2], torch.sin(norm.unsqueeze(-1) * 1000.0 * div), atol=1e-6)
    assert torch.allclose(got[..., 1::2], torch.cos(norm.unsqueeze(-1) * 1000.0 * div), atol=1e-6)


def _canonical_model() -> LIBSTransformer:
    meta = spectral_meta_from_config(
        {"patch": {**CFG, "preprocess": "canonical", "input_scale": 1.0}}, WL
    )
    return LIBSTransformer(
        n_bins=WL.size, d_model=16, n_heads=2, n_layers=1, d_ff=32, dropout=0.0,
        embedding_type="spectral_patch", spectral_meta=meta,
    )  # fmt: skip


def test_canonical_mode_takes_values_and_mask():
    m = _canonical_model()
    x = torch.rand(2, 2 * WL.size)
    x[:, WL.size :] = (x[:, WL.size :] > 0.99).float()  # a saturation mask half
    out = m(x)
    assert out["mip_predictions"].shape == (2, m.embedding.n_tokens, CFG["n_samples"])
    with pytest.raises(ValueError, match="canonical input"):
        m(torch.rand(2, WL.size))  # unit-normalised input is rejected loudly
    assert "canonical" not in spectral_meta_from_config(
        {"patch": {"canonical": {"k_sigma": 3}}}, WL
    )


def test_canonical_flags_are_the_mask_and_masking_hides_both_halves():
    emb = _canonical_model().embedding
    emb.eval()
    x = torch.zeros(1, 2 * WL.size)
    x[0, : WL.size] = 0.3
    base = emb(x)
    y = x.clone()
    y[0, WL.size + 100 : WL.size + 110] = 1.0  # flags set, values unchanged
    assert not torch.equal(base, emb(y))  # the mask half reaches the tokens
    tm = torch.zeros(1, emb.n_tokens, dtype=torch.bool)
    tm[0, emb.n_tokens // 2] = True
    hidden = emb.px_stitched[emb.pixel_mask(tm)[0]]
    a, b = x.clone(), x.clone()
    b[0, hidden] = 0.9  # hidden values
    b[0, WL.size + hidden] = 1.0  # hidden flags
    assert torch.equal(emb(a, token_mask=tm), emb(b, token_mask=tm))


def test_canonical_run_info_and_old_checkpoints():
    m = _canonical_model()
    info = spectral_patch_run_info(
        spectral_meta_from_config({"patch": {**CFG, "preprocess": "canonical"}}, WL), m, None,
        input_spec={"preprocess": "canonical", "canonical": {"k_sigma": 3.0, "blank_nm": [[265.4, 267.0]]}},
        input_units="full_well",
    )  # fmt: skip
    assert info["preprocess"] == "canonical" and info["input_units"] == "full_well"
    assert info["canonical"]["blank_nm"] == [[265.4, 267.0]]
    old = _model()  # a 'none' model: same parameters, so its state_dict loads strictly
    m.load_state_dict(old.state_dict(), strict=True)
