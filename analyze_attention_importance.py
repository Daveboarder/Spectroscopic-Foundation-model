"""
Attention-based input-importance for fine-tuned LIBS models.

Loads a fine-tuned run (primarily `quantification_binned` with the
`line_token_linear` embedding), runs the test spectra through the encoder while
collecting CLS-token attention, and reports:

  1. Per-line / per-bin importance  — how much the CLS token attends to each
     spectral line (or wavelength bin), aggregated over heads, layers and a
     sample of test spectra. Optionally grouped by element (atomic number).

  2. Per-channel importance for the 14 named token features — attention cannot
     decompose channels by itself (they are fused by the embedding's
     `nn.Linear(14, d_model)` before attention), so channel importance is
     attention-informed: it combines the CLS attention placed on each line with
     the z-scored feature magnitude and the projection column norm.

  3. Mutual line-line relationships from token self-attention — which lines
     exchange information with which other lines. Two views:
       a) top-K x top-K line-pair attention matrix (top lines by CLS
          importance), head- and layer-averaged, plus the same matrix for
          each attention head separately (layer-averaged);
       b) element-to-element attention matrix: average attention a line of
          element A pays to the lines of element B, aggregated over all lines.
     High off-diagonal weight between lines of the SAME element suggests the
     model cross-references multiple lines of one species (consistency /
     robustness to single-line distortion); weight between DIFFERENT elements
     points to learned co-occurrence in the sample matrix or to spectral
     proximity / interference between those lines.

Scope / caveats (also written to the summary file):
  - Attention reflects the *shared* CLS representation. The binned head's
    per-element branches all read the same pooled vector, so attention is not a
    per-output-element attribution; that would need a gradient-based method.
  - The per-channel breakdown relies on the linear embedding and is only
    produced for `line_token_linear` runs. `intensity` (bin) runs get per-bin
    importance only.

Usage:
    uv run python analyze_attention_importance.py \
        --run_dir runs/finetune_2026-06-04_21-03-15_libs_binned_ft

    # Lightweight path: sample spectra straight from the token cache instead of
    # rebuilding the full labeled dataset / split.
    uv run python analyze_attention_importance.py \
        --run_dir runs/finetune_... --tokens_path external_data/cache/line_tokens_<hash>.h5
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from data.line_tokenization import F_Z, FEATURE_NAMES
from models.libs_transformer import LIBSTransformer
from utils.run_manager import RunManager

# Reuse loading / data helpers so the analyzed split matches training exactly.
from train_finetune import (
    _load_weights_shape_safe,
    generate_labeled_data,
)
from data.line_embedding_pipeline import prepare_line_tokens_assets


# Atomic-number -> element symbol (enough to cover the LIBS line dictionary).
_ELEMENT_SYMBOLS = [
    "n", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg",
    "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn",
    "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb",
    "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In",
    "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm",
    "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta",
    "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At",
    "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",
]


def _z_to_symbol(z: int) -> str:
    if 0 <= z < len(_ELEMENT_SYMBOLS):
        return _ELEMENT_SYMBOLS[z]
    return f"Z{z}"


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ────────────────────────────────────────────────────────────────────────────
# Model + data loading
# ────────────────────────────────────────────────────────────────────────────

def _checkpoint_encoder_state(checkpoint_path: str) -> dict:
    """Extract the `encoder.*` sub-state-dict from a finetune Lightning ckpt."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    encoder_state = {
        k[len("encoder."):]: v for k, v in state_dict.items() if k.startswith("encoder.")
    }
    if not encoder_state:
        # Fall back to a raw encoder state dict (final_encoder.pt / encoder_latest.pt).
        encoder_state = dict(state_dict)
    # Pretrain MIP head is unused here and may mismatch; drop it.
    encoder_state = {k: v for k, v in encoder_state.items() if not k.startswith("mip_head.")}
    return encoder_state


def build_encoder(
    config: dict, run_info: dict, token_meta: dict, wavelength=None,
) -> LIBSTransformer:
    """Build a LIBSTransformer matching the fine-tuned encoder (no task heads).

    ``spectral_patch`` encoders are rebuilt on ``wavelength`` (nm) when given
    (e.g. the axis of a measured map), otherwise on the training axis recorded
    in ``run_info.spectral_patch.axis``."""
    emb_type = run_info.get("embedding_type") or config["model"].get("embedding_type", "intensity")
    kwargs = dict(
        n_bins=config["data"]["n_bins"],
        d_model=config["model"]["d_model"],
        n_heads=config["model"]["n_heads"],
        n_layers=config["model"]["n_layers"],
        d_ff=config["model"]["d_ff"],
        dropout=0.0,
        n_classes=config["data"]["n_classes"],
        embedding_type=emb_type,
    )
    if emb_type == "line_token_linear":
        if token_meta is None:
            raise ValueError("line_token_linear encoder requires token_meta")
        kwargs["n_lines"] = token_meta["n_lines"]
        kwargs["line_token_meta"] = token_meta
        kwargs["n_mip_target_channels"] = int(config["model"].get("n_mip_target_channels", 2))
    elif emb_type == "spectral_patch":
        from models.spectral_patch_embedding import (
            DEFAULT_PATCH_CFG,
            spectral_meta_from_config,
            wavelength_from_run_info,
        )

        info = run_info.get("spectral_patch") or {}
        wl = wavelength_from_run_info(run_info) if wavelength is None else wavelength
        # the window settings training used (recorded in run_info) win over the
        # run's config.yaml, which is a copy of the fine-tune --config
        recorded = {k: info[k] for k in DEFAULT_PATCH_CFG if k in info}
        model_cfg = {
            **config["model"],
            "patch": {**(config["model"].get("patch") or {}), **recorded},
        }
        kwargs["spectral_meta"] = spectral_meta_from_config(
            model_cfg, wl, n_segments=info.get("n_segments"),
        )
        kwargs["n_bins"] = len(wl)
    elif emb_type == "line_token":
        raise NotImplementedError(
            "line_token (runtime embedding) is not supported by this analyzer; "
            "use a line_token_linear run, or extend build_encoder with line_dict_meta."
        )
    model = LIBSTransformer(**kwargs)
    if emb_type == "spectral_patch" and wavelength is None and info.get("n_tokens"):
        if model.embedding.n_tokens != int(info["n_tokens"]):
            raise ValueError(
                f"rebuilt spectral_patch embedding has {model.embedding.n_tokens} windows, "
                f"training had {info['n_tokens']}"
            )
    return model


def prepare_data(config: dict, run_info: dict, args):
    """Return (dataset, token_meta, source_description).

    Two paths:
      * tokens cache directly (lightweight, label-free) when --tokens_path is
        given or --use_token_cache is set with a path available from run_info;
      * full pipeline reconstruction (faithful test split) otherwise.
    """
    from data.dataset import LineTokensLabeledDataset, LineTokensDataset

    line_embedding_config = run_info.get("line_embedding_config")
    libs_data_config = run_info.get("libs_data_config")
    seed = int(run_info.get("seed", 42))

    tokens_path = args.tokens_path or (
        run_info.get("line_tokens_path") if args.use_token_cache else None
    )

    if tokens_path:
        tp = Path(tokens_path)
        if not tp.is_file():
            raise FileNotFoundError(
                f"tokens cache not found: {tokens_path}. Drop --tokens_path/"
                "--use_token_cache to rebuild from the pipeline instead."
            )
        import h5py
        with h5py.File(tp, "r") as f:
            n_spectra = int(f.attrs.get("n_spectra", f["tokens"].shape[0]))
            feature_mean = np.asarray(f.attrs["feature_mean"], dtype=np.float32)
            feature_std = np.asarray(f.attrs["feature_std"], dtype=np.float32)
            central_wavelength = f["central_wavelength"][:].astype(np.float32)
            n_lines = int(f.attrs["n_lines"])
            n_features = int(f.attrs["n_features"])
        rng = np.random.default_rng(seed)
        n_take = min(args.max_samples, n_spectra)
        indices = np.sort(rng.choice(n_spectra, size=n_take, replace=False))
        dataset = LineTokensDataset(str(tp), indices=indices)
        token_meta = {
            "n_lines": n_lines,
            "n_features": n_features,
            "feature_names": FEATURE_NAMES,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "central_wavelength": central_wavelength,
        }
        return dataset, token_meta, f"token cache {tp.name} ({n_take} spectra)"

    # ── Faithful reconstruction of the trained test split ──
    if not (line_embedding_config and libs_data_config):
        raise ValueError(
            "run_info is missing line_embedding_config / libs_data_config; "
            "pass --tokens_path to analyze a token cache directly instead."
        )
    data = generate_labeled_data(config, seed=seed, libs_config_path=libs_data_config)
    ds = data["libs_dataset"]
    token_meta = prepare_line_tokens_assets(
        ds.spectra.astype(np.float32),
        ds.wavelength,
        line_embedding_config,
        spectra_cache_key=ds.cache_key,
        verbose=True,
    )
    tokens_path = token_meta["line_tokens_path"]
    config["data"]["n_bins"] = token_meta["n_lines"]
    config["model"]["max_seq_len"] = token_meta["n_lines"] + 1

    test_indices = np.asarray(data["splits"]["test"], dtype=np.int64)
    _, test_labels, test_conc = data["test"]
    rng = np.random.default_rng(seed)
    if args.max_samples < len(test_indices):
        pick = np.sort(rng.choice(len(test_indices), size=args.max_samples, replace=False))
        test_indices = test_indices[pick]
        test_labels = test_labels[pick]
        test_conc = test_conc[pick]

    dataset = LineTokensLabeledDataset(
        tokens_path, test_labels, concentrations=test_conc, indices=test_indices,
    )
    return dataset, token_meta, f"reconstructed test split ({len(test_indices)} spectra)"


# ────────────────────────────────────────────────────────────────────────────
# Attention extraction (bounded memory)
# ────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def cls_attention_per_layer(encoder: LIBSTransformer, inputs: dict):
    """Run the encoder block loop manually, keeping only the CLS attention row.

    Returns:
        cls_rows: list of [B, S] head-averaged CLS->position attention per layer
        kpm: [B, S] key_padding_mask (True = ignore) or None
    """
    emb_type = encoder.embedding_type
    if emb_type == "line_token_linear":
        hidden, kpm = encoder.embedding(inputs["tokens"], fit_valid=inputs.get("fit_valid"))
    elif emb_type == "line_token":
        hidden, kpm = encoder.embedding(inputs["line_features"])
    else:
        hidden = encoder.embedding(inputs["spectrum"], mask=None)
        kpm = None

    cls_rows = []
    for block in encoder.encoder_blocks:
        hidden, attn = block(hidden, key_padding_mask=kpm, need_weights=True)
        # attn: [B, n_heads, S, S]; take CLS query row (index 0), average heads.
        cls_rows.append(attn[:, :, 0, :].mean(dim=1).detach())
        del attn
    return cls_rows, kpm


def _normalize_line_attention(cls_rows, kpm):
    """Aggregate per-layer CLS rows to per-line attention.

    Returns dict of [B, n_lines] tensors for 'layer_mean' and 'last_layer',
    each masked to valid lines and renormalized to sum to 1 per spectrum.
    """
    stack = torch.stack(cls_rows, dim=0)        # [L, B, S]
    layer_mean = stack.mean(dim=0)              # [B, S]
    last_layer = cls_rows[-1]                   # [B, S]

    def _lines(a):
        lines = a[:, 1:]                        # drop CLS column -> [B, n_lines]
        if kpm is not None:
            valid = (~kpm[:, 1:]).to(lines.dtype)
            lines = lines * valid
        denom = lines.sum(dim=1, keepdim=True).clamp(min=1e-12)
        return lines / denom

    return {"layer_mean": _lines(layer_mean), "last_layer": _lines(last_layer)}


def _drop_zero_valid_spectra(batch):
    """Drop spectra whose Voigt fits all failed (fit_valid all zero).

    For such spectra the embedding masks every attention key (the CLS key is
    masked unconditionally by `LinearLineTokenEmbedding`), so the softmax over
    an all-masked row returns NaN and would poison the running importance sums.

    Returns (filtered_batch_or_None, n_dropped).
    """
    if "fit_valid" not in batch:
        return batch, 0
    keep = batch["fit_valid"].sum(dim=1) > 0
    n_dropped = int((~keep).sum().item())
    if n_dropped == 0:
        return batch, 0
    if not keep.any():
        return None, n_dropped
    return {k: v[keep] for k, v in batch.items()}, n_dropped


def _build_inputs(batch, emb_type, device):
    inputs = {}
    if emb_type == "line_token_linear":
        inputs["tokens"] = batch["tokens"].to(device)
        if "fit_valid" in batch:
            inputs["fit_valid"] = batch["fit_valid"].to(device)
    elif emb_type == "line_token":
        inputs["line_features"] = batch["line_features"].to(device)
    else:
        inputs["spectrum"] = batch["spectrum"].to(device)
    return inputs


@torch.no_grad()
def mutual_attention_pass(encoder, loader, top_idx, atomic_numbers, args):
    """Second pass: line-to-line self-attention relationships.

    Collects, layer-averaged over `args.pair_samples` spectra:
      * pair_heads [H, K, K]: attention among the top-K lines per attention
        head (queries = rows, keys = columns). Values are raw attention
        weights (each full row over all sequence positions sums to 1, so
        sub-matrix rows do not).
      * pair_mean  [K, K]: the same matrix averaged over heads.
      * elem_mean  [E, E]: mean attention a *valid* line of element A (query)
        pays to all lines of element B (keys), aggregated over every line
        (head-averaged).

    Returns (pair_mean [K,K] np, pair_heads [H,K,K] np, elem_symbols list,
    elem_mean [E,E] np).
    """
    device = args.device
    emb_type = encoder.embedding_type
    n_layers = len(encoder.encoder_blocks)
    n_heads = encoder.encoder_blocks[0].self_attn.num_heads
    top_idx_t = torch.as_tensor(np.ascontiguousarray(top_idx), dtype=torch.long, device=device)
    K = len(top_idx)

    pair_sum_heads = torch.zeros(n_heads, K, K, dtype=torch.float64)

    # Element grouping (one-hot over atomic numbers present in the dictionary).
    if atomic_numbers is not None:
        uniq_z = np.unique(atomic_numbers)
        z_to_col = {int(z): i for i, z in enumerate(uniq_z)}
        n_elem = len(uniq_z)
        G = torch.zeros(len(atomic_numbers), n_elem, device=device)
        for i, z in enumerate(atomic_numbers):
            G[i, z_to_col[int(z)]] = 1.0
        elem_sum = torch.zeros(n_elem, n_elem, dtype=torch.float64)
        elem_symbols = [_z_to_symbol(int(z)) for z in uniq_z]
    else:
        G = None
        elem_sum = None
        elem_symbols = []

    n_seen = 0
    n_skipped = 0
    print(f"\nRunning mutual (line-line) attention pass on up to "
          f"{args.pair_samples} spectra...")
    for batch in loader:
        if n_seen >= args.pair_samples:
            break
        batch, dropped = _drop_zero_valid_spectra(batch)
        n_skipped += dropped
        if batch is None:
            continue
        inputs = _build_inputs(batch, emb_type, device)

        if emb_type == "line_token_linear":
            hidden, kpm = encoder.embedding(inputs["tokens"], fit_valid=inputs.get("fit_valid"))
        elif emb_type == "line_token":
            hidden, kpm = encoder.embedding(inputs["line_features"])
        else:
            hidden = encoder.embedding(inputs["spectrum"], mask=None)
            kpm = None

        B = hidden.shape[0]
        if kpm is not None:
            valid = (~kpm[:, 1:]).float()                     # [B, L]
        else:
            valid = torch.ones(B, hidden.shape[1] - 1, device=device)

        for block in encoder.encoder_blocks:
            hidden, attn = block(hidden, key_padding_mask=kpm, need_weights=True)

            # Per-head top-K submatrix (CLS at index 0 -> shift line indices).
            sub = attn.index_select(2, top_idx_t + 1).index_select(3, top_idx_t + 1)
            # Zero rows of invalid (padded) query lines; invalid keys are
            # already ~0 from the softmax key_padding_mask.
            valid_top = valid.index_select(1, top_idx_t)          # [B, K]
            sub = sub * valid_top.unsqueeze(1).unsqueeze(-1)      # [B, H, K, K]
            pair_sum_heads += sub.sum(dim=0).double().cpu()

            # [B, H, S, S] -> head mean, drop CLS row/col -> [B, L, L]
            A = attn.mean(dim=1)[:, 1:, 1:]
            del attn
            A = A * valid.unsqueeze(-1)

            if G is not None:
                T = torch.matmul(A, G)                        # [B, L, E] sum over keys in B
                M = torch.einsum("la,ble->bae", G, T)         # [B, E, E] sum over queries in A
                q_count = valid @ G                           # [B, E] valid queries per element
                M = M / q_count.clamp(min=1.0).unsqueeze(-1)
                elem_sum += M.sum(dim=0).double().cpu()
            del A

        n_seen += B

    denom = max(n_seen, 1) * n_layers
    pair_heads = (pair_sum_heads / denom).numpy()
    pair_mean = pair_heads.mean(axis=0)
    elem_mean = (elem_sum / denom).numpy() if elem_sum is not None else None
    if n_skipped:
        print(f"  skipped {n_skipped} spectra with zero valid lines")
    print(f"  aggregated over {n_seen} spectra x {n_layers} layers x {n_heads} heads")
    return pair_mean, pair_heads, elem_symbols, elem_mean


# ────────────────────────────────────────────────────────────────────────────
# Main analysis
# ────────────────────────────────────────────────────────────────────────────

def run_analysis(encoder, dataset, token_meta, config, run_info, args, output_dir: Path):
    device = args.device
    encoder = encoder.to(device).eval()
    emb_type = encoder.embedding_type

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    )

    n_lines = int(token_meta["n_lines"])
    central_wavelength = np.asarray(token_meta["central_wavelength"], dtype=np.float32)

    # Per-channel pieces (line_token_linear only).
    do_channels = emb_type == "line_token_linear"
    if do_channels:
        feature_mean = encoder.embedding.feature_mean.to(device)
        feature_std = encoder.embedding.feature_std.to(device)
        W = encoder.embedding.projection.weight.detach()      # [d_model, n_features]
        col_norm = W.norm(dim=0).cpu().numpy()                # [n_features]
        n_features = W.shape[1]
        chan_importance = np.zeros(n_features, dtype=np.float64)
        chan_absz_sum = np.zeros(n_features, dtype=np.float64)
        chan_valid_count = 0.0

    line_importance = np.zeros(n_lines, dtype=np.float64)
    line_importance_last = np.zeros(n_lines, dtype=np.float64)
    atomic_numbers = None
    n_seen = 0
    n_skipped = 0

    print(f"\nRunning attention pass on {len(dataset)} spectra "
          f"(batch_size={args.batch_size}, device={device})...")
    for bi, batch in enumerate(loader):
        if atomic_numbers is None and "tokens" in batch:
            # Atomic number is a static per-line dictionary value (channel F_Z);
            # grab it before any filtering so it is set even if the first batch
            # is dropped entirely.
            atomic_numbers = np.rint(batch["tokens"][0, :, F_Z].cpu().numpy()).astype(int)

        batch, dropped = _drop_zero_valid_spectra(batch)
        n_skipped += dropped
        if batch is None:
            continue

        inputs = _build_inputs(batch, emb_type, device)
        cls_rows, kpm = cls_attention_per_layer(encoder, inputs)
        agg = _normalize_line_attention(cls_rows, kpm)
        a_mean = agg["layer_mean"]              # [B, n_lines]
        a_last = agg["last_layer"]

        line_importance += a_mean.sum(dim=0).cpu().numpy()
        line_importance_last += a_last.sum(dim=0).cpu().numpy()

        if do_channels:
            tokens = inputs["tokens"]
            z = (tokens - feature_mean) / feature_std        # [B, n_lines, F]
            absz = z.abs()
            # I_c contribution = sum_lines a_i * |z_i,c|, weighted by col_norm later.
            contrib = torch.einsum("bn,bnc->bc", a_mean, absz)   # [B, F]
            chan_importance += contrib.sum(dim=0).cpu().numpy()
            # mean|z| diagnostic over valid lines only.
            if kpm is not None:
                valid = (~kpm[:, 1:]).unsqueeze(-1).to(absz.dtype)
                chan_absz_sum += (absz * valid).sum(dim=(0, 1)).cpu().numpy()
                chan_valid_count += float(valid.sum().item())
            else:
                chan_absz_sum += absz.sum(dim=(0, 1)).cpu().numpy()
                chan_valid_count += float(absz.shape[0] * absz.shape[1])

        n_seen += a_mean.shape[0]
        if (bi + 1) % 20 == 0:
            print(f"  processed {n_seen}/{len(dataset)} spectra")

    if n_skipped:
        print(f"  skipped {n_skipped} spectra with zero valid lines "
              f"(all Voigt fits failed; attention would be NaN)")
    line_importance /= max(n_seen, 1)
    line_importance_last /= max(n_seen, 1)
    if do_channels:
        chan_importance = (chan_importance / max(n_seen, 1)) * col_norm
        mean_absz = chan_absz_sum / max(chan_valid_count, 1.0)

    # ── Per-line CSV ──
    order = np.argsort(line_importance)[::-1]
    per_line_csv = output_dir / "per_line_importance.csv"
    with open(per_line_csv, "w") as f:
        f.write("rank,line_index,central_wavelength_nm,atomic_number,element,"
                "importance_layer_mean,importance_last_layer\n")
        for rank, idx in enumerate(order):
            z = int(atomic_numbers[idx]) if atomic_numbers is not None else -1
            sym = _z_to_symbol(z) if z >= 0 else "?"
            wl = float(central_wavelength[idx]) if idx < len(central_wavelength) else float("nan")
            f.write(f"{rank},{idx},{wl:.4f},{z},{sym},"
                    f"{line_importance[idx]:.8e},{line_importance_last[idx]:.8e}\n")
    print(f"Saved: {per_line_csv}")

    # ── Per-element grouping ──
    element_importance = {}
    if atomic_numbers is not None:
        for idx in range(n_lines):
            sym = _z_to_symbol(int(atomic_numbers[idx]))
            element_importance[sym] = element_importance.get(sym, 0.0) + float(line_importance[idx])
        elem_sorted = sorted(element_importance.items(), key=lambda kv: kv[1], reverse=True)
        per_elem_csv = output_dir / "per_element_importance.csv"
        with open(per_elem_csv, "w") as f:
            f.write("element,summed_importance,n_lines\n")
            counts = {}
            for idx in range(n_lines):
                sym = _z_to_symbol(int(atomic_numbers[idx]))
                counts[sym] = counts.get(sym, 0) + 1
            for sym, val in elem_sorted:
                f.write(f"{sym},{val:.8e},{counts.get(sym, 0)}\n")
        print(f"Saved: {per_elem_csv}")
    else:
        elem_sorted = []

    # ── Per-channel CSV ──
    if do_channels:
        feature_names = list(token_meta.get("feature_names", FEATURE_NAMES))
        chan_norm = chan_importance / max(chan_importance.sum(), 1e-12)
        chan_order = np.argsort(chan_importance)[::-1]
        per_chan_csv = output_dir / "per_channel_importance.csv"
        with open(per_chan_csv, "w") as f:
            f.write("rank,channel_index,channel_name,importance,importance_fraction,"
                    "projection_col_norm,mean_abs_zscore\n")
            for rank, c in enumerate(chan_order):
                name = feature_names[c] if c < len(feature_names) else f"feat_{c}"
                f.write(f"{rank},{c},{name},{chan_importance[c]:.8e},{chan_norm[c]:.6f},"
                        f"{col_norm[c]:.6f},{mean_absz[c]:.6f}\n")
        print(f"Saved: {per_chan_csv}")

    # ── Mutual line-line attention (second pass over the same loader) ──
    pair_data = None
    if not args.skip_pairs:
        top_idx = order[: min(args.pair_top_k, n_lines)].copy()
        pair_mean, pair_heads, elem_symbols, elem_mean = mutual_attention_pass(
            encoder, loader, top_idx, atomic_numbers, args,
        )
        pair_labels = []
        for idx in top_idx:
            sym = _z_to_symbol(int(atomic_numbers[idx])) if atomic_numbers is not None else "?"
            pair_labels.append(f"{sym} {float(central_wavelength[idx]):.1f}")
        pair_data = {
            "top_idx": top_idx,
            "pair_mean": pair_mean,
            "pair_heads": pair_heads,
            "pair_labels": pair_labels,
            "elem_symbols": elem_symbols,
            "elem_mean": elem_mean,
        }

        # line-pair CSV (directional, self-attention diagonal excluded)
        pair_csv = output_dir / "line_pair_attention.csv"
        K = len(top_idx)
        pairs = [
            (pair_mean[i, j], i, j)
            for i in range(K) for j in range(K) if i != j
        ]
        pairs.sort(reverse=True)
        with open(pair_csv, "w") as f:
            f.write("rank,query_line_index,query_element,query_wavelength_nm,"
                    "key_line_index,key_element,key_wavelength_nm,attention\n")
            for rank, (val, i, j) in enumerate(pairs):
                qi, kj = int(top_idx[i]), int(top_idx[j])
                qz = int(atomic_numbers[qi]) if atomic_numbers is not None else -1
                kz = int(atomic_numbers[kj]) if atomic_numbers is not None else -1
                f.write(f"{rank},{qi},{_z_to_symbol(qz) if qz >= 0 else '?'},"
                        f"{float(central_wavelength[qi]):.4f},"
                        f"{kj},{_z_to_symbol(kz) if kz >= 0 else '?'},"
                        f"{float(central_wavelength[kj]):.4f},{val:.8e}\n")
        print(f"Saved: {pair_csv}")

        # per-head line-pair CSV (directional, self-attention diagonal excluded)
        pair_heads_csv = output_dir / "line_pair_attention_per_head.csv"
        with open(pair_heads_csv, "w") as f:
            f.write("head,rank,query_line_index,query_element,query_wavelength_nm,"
                    "key_line_index,key_element,key_wavelength_nm,attention\n")
            for h in range(pair_heads.shape[0]):
                h_pairs = [
                    (pair_heads[h, i, j], i, j)
                    for i in range(K) for j in range(K) if i != j
                ]
                h_pairs.sort(reverse=True)
                for rank, (val, i, j) in enumerate(h_pairs):
                    qi, kj = int(top_idx[i]), int(top_idx[j])
                    qz = int(atomic_numbers[qi]) if atomic_numbers is not None else -1
                    kz = int(atomic_numbers[kj]) if atomic_numbers is not None else -1
                    f.write(f"{h},{rank},{qi},{_z_to_symbol(qz) if qz >= 0 else '?'},"
                            f"{float(central_wavelength[qi]):.4f},"
                            f"{kj},{_z_to_symbol(kz) if kz >= 0 else '?'},"
                            f"{float(central_wavelength[kj]):.4f},{val:.8e}\n")
        print(f"Saved: {pair_heads_csv}")

        # element-element matrix CSV
        if elem_mean is not None:
            elem_csv = output_dir / "element_attention_matrix.csv"
            with open(elem_csv, "w") as f:
                f.write("query_element," + ",".join(elem_symbols) + "\n")
                for a, sym in enumerate(elem_symbols):
                    f.write(sym + "," + ",".join(f"{elem_mean[a, b]:.8e}"
                                                 for b in range(len(elem_symbols))) + "\n")
            print(f"Saved: {elem_csv}")

    # ── Plots ──
    _plot_line_vs_wavelength(central_wavelength, line_importance, atomic_numbers, output_dir)
    if elem_sorted:
        _plot_per_element(elem_sorted, output_dir)
    if do_channels:
        _plot_per_channel(
            [token_meta.get("feature_names", FEATURE_NAMES)[c] for c in chan_order],
            chan_importance[chan_order], output_dir,
        )
    if pair_data is not None:
        _plot_pair_heatmap(pair_data["pair_labels"], pair_data["pair_mean"], output_dir)
        _plot_pair_heads(pair_data["pair_labels"], pair_data["pair_heads"], output_dir)
        if pair_data["elem_mean"] is not None:
            _plot_element_attention(pair_data["elem_symbols"], pair_data["elem_mean"],
                                    output_dir)

    # ── Summary ──
    _write_summary(
        output_dir, run_info, config, args, n_seen, order, line_importance,
        central_wavelength, atomic_numbers, elem_sorted,
        chan_order if do_channels else None,
        chan_importance if do_channels else None,
        token_meta.get("feature_names", FEATURE_NAMES) if do_channels else None,
        emb_type,
        pair_data=pair_data,
    )


def _plot_line_vs_wavelength(wavelength, importance, atomic_numbers, output_dir, top_k=15):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(wavelength, importance, lw=0.6, color="steelblue", alpha=0.8)
    top = np.argsort(importance)[::-1][:top_k]
    ax.scatter(wavelength[top], importance[top], color="crimson", s=20, zorder=5)
    for idx in top:
        sym = _z_to_symbol(int(atomic_numbers[idx])) if atomic_numbers is not None else ""
        ax.annotate(f"{sym} {wavelength[idx]:.1f}", (wavelength[idx], importance[idx]),
                    textcoords="offset points", xytext=(0, 6), fontsize=7, rotation=45,
                    ha="left")
    ax.set_xlabel("Central wavelength (nm)")
    ax.set_ylabel("Mean CLS attention importance")
    ax.set_title("Per-line attention importance vs wavelength (top lines annotated)")
    fig.tight_layout()
    fig.savefig(output_dir / "per_line_importance_vs_wavelength.png", dpi=150)
    plt.close(fig)
    print("Saved: per_line_importance_vs_wavelength.png")


def _plot_per_element(elem_sorted, output_dir, top_k=25):
    items = elem_sorted[:top_k]
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(names)), vals, color="seagreen", edgecolor="black")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Summed attention importance")
    ax.set_title(f"Attention importance grouped by element (top {len(names)})")
    fig.tight_layout()
    fig.savefig(output_dir / "per_element_importance.png", dpi=150)
    plt.close(fig)
    print("Saved: per_element_importance.png")


def _plot_per_channel(names, vals, output_dir):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(range(len(names)), vals, color="indigo", edgecolor="black")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Attention-weighted importance")
    ax.set_title("Per-channel (token feature) importance")
    fig.tight_layout()
    fig.savefig(output_dir / "per_channel_importance.png", dpi=150)
    plt.close(fig)
    print("Saved: per_channel_importance.png")


def _plot_pair_heatmap(labels, matrix, output_dir):
    K = len(labels)
    fig, ax = plt.subplots(figsize=(max(8, 0.28 * K + 3), max(7, 0.28 * K + 2)))
    im = ax.imshow(matrix, cmap="viridis", aspect="equal")
    ax.set_xticks(range(K))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(range(K))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Key line (attended to)")
    ax.set_ylabel("Query line (attending)")
    ax.set_title("Line-line self-attention among top lines\n"
                 "(head- and layer-averaged; rows = queries)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="attention weight")
    fig.tight_layout()
    fig.savefig(output_dir / "line_pair_attention_heatmap.png", dpi=150)
    plt.close(fig)
    print("Saved: line_pair_attention_heatmap.png")


def _plot_pair_heads(labels, pair_heads, output_dir):
    """Grid of per-head line-line attention heatmaps (shared color scale)."""
    H, K, _ = pair_heads.shape
    ncols = min(4, H)
    nrows = (H + ncols - 1) // ncols
    panel = max(3.2, 0.11 * K + 1.5)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(panel * ncols + 1.5, panel * nrows + 1),
                             squeeze=False)
    vmin = float(pair_heads.min())
    vmax = float(pair_heads.max())
    im = None
    for h in range(nrows * ncols):
        ax = axes[h // ncols][h % ncols]
        if h >= H:
            ax.axis("off")
            continue
        im = ax.imshow(pair_heads[h], cmap="viridis", aspect="equal",
                       vmin=vmin, vmax=vmax)
        ax.set_title(f"head {h}", fontsize=9)
        ax.set_xticks(range(K))
        ax.set_yticks(range(K))
        if h % ncols == 0:
            ax.set_yticklabels(labels, fontsize=4)
        else:
            ax.set_yticklabels([])
        if h // ncols == nrows - 1 or h + ncols >= H:
            ax.set_xticklabels(labels, rotation=90, fontsize=4)
        else:
            ax.set_xticklabels([])
    fig.suptitle("Line-line self-attention per head\n"
                 "(layer-averaged; rows = queries, columns = keys; shared color scale)",
                 fontsize=11)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="attention weight")
    fig.savefig(output_dir / "line_pair_attention_per_head.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("Saved: line_pair_attention_per_head.png")


def _plot_element_attention(elem_symbols, matrix, output_dir):
    E = len(elem_symbols)
    fig, ax = plt.subplots(figsize=(max(9, 0.25 * E + 3), max(8, 0.25 * E + 2)))
    # Log color scale: element-pair attention spans orders of magnitude.
    vals = np.maximum(matrix, 1e-12)
    im = ax.imshow(np.log10(vals), cmap="magma", aspect="equal")
    ax.set_xticks(range(E))
    ax.set_xticklabels(elem_symbols, rotation=90, fontsize=7)
    ax.set_yticks(range(E))
    ax.set_yticklabels(elem_symbols, fontsize=7)
    ax.set_xlabel("Key element (lines attended to)")
    ax.set_ylabel("Query element (lines attending)")
    ax.set_title("Element-to-element self-attention\n"
                 "(mean attention per query line, log10 scale)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="log10(mean attention)")
    fig.tight_layout()
    fig.savefig(output_dir / "element_attention_heatmap.png", dpi=150)
    plt.close(fig)
    print("Saved: element_attention_heatmap.png")


def _write_summary(output_dir, run_info, config, args, n_seen, order, line_importance,
                   wavelength, atomic_numbers, elem_sorted, chan_order, chan_importance,
                   feature_names, emb_type, pair_data=None):
    path = output_dir / "importance_summary.txt"
    with open(path, "w") as f:
        f.write("Attention-based input importance\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Run: {run_info.get('run_name', '?')}\n")
        f.write(f"Task: {run_info.get('task', '?')}    Pool: {run_info.get('pool', '?')}\n")
        f.write(f"Embedding: {emb_type}\n")
        f.write(f"Spectra analyzed: {n_seen}\n\n")

        f.write("Top 20 lines (by mean CLS attention):\n")
        f.write("-" * 50 + "\n")
        for rank, idx in enumerate(order[:20]):
            z = int(atomic_numbers[idx]) if atomic_numbers is not None else -1
            sym = _z_to_symbol(z) if z >= 0 else "?"
            wl = float(wavelength[idx]) if idx < len(wavelength) else float("nan")
            f.write(f"  {rank + 1:2d}. {sym:>3s} {wl:8.2f} nm   "
                    f"importance={line_importance[idx]:.4e}\n")

        if elem_sorted:
            f.write("\nTop 15 elements (summed line importance):\n")
            f.write("-" * 50 + "\n")
            for sym, val in elem_sorted[:15]:
                f.write(f"  {sym:>3s}   {val:.4e}\n")

        if chan_order is not None:
            f.write("\nPer-channel importance (all 14 token features):\n")
            f.write("-" * 50 + "\n")
            total = max(chan_importance.sum(), 1e-12)
            for c in chan_order:
                name = feature_names[c] if c < len(feature_names) else f"feat_{c}"
                f.write(f"  {name:>28s}   {chan_importance[c]:.4e}   "
                        f"({100 * chan_importance[c] / total:5.1f}%)\n")

        if pair_data is not None:
            pair_mean = pair_data["pair_mean"]
            top_idx = pair_data["top_idx"]
            K = len(top_idx)

            f.write("\nMutual line-line relationships (token self-attention):\n")
            f.write("-" * 50 + "\n")
            f.write("Attention weight q->k measures how strongly the representation\n")
            f.write("of line q pulls information from line k when the encoder builds\n")
            f.write("its contextual embedding (head- and layer-averaged, directional).\n\n")

            pairs = [
                (float(pair_mean[i, j]), i, j)
                for i in range(K) for j in range(K) if i != j
            ]
            pairs.sort(reverse=True)
            f.write(f"Top 15 line pairs among the {K} most important lines:\n")
            for rank, (val, i, j) in enumerate(pairs[:15]):
                qi, kj = int(top_idx[i]), int(top_idx[j])
                qz = int(atomic_numbers[qi]) if atomic_numbers is not None else -1
                kz = int(atomic_numbers[kj]) if atomic_numbers is not None else -1
                qs = _z_to_symbol(qz) if qz >= 0 else "?"
                ks = _z_to_symbol(kz) if kz >= 0 else "?"
                same = "same element" if (qz == kz and qz >= 0) else "cross-element"
                f.write(f"  {rank + 1:2d}. {qs:>3s} {float(wavelength[qi]):8.2f} nm "
                        f"-> {ks:>3s} {float(wavelength[kj]):8.2f} nm   "
                        f"attn={val:.4e}   ({same})\n")

            pair_heads = pair_data.get("pair_heads")
            if pair_heads is not None:
                f.write("\nPer-head specialization (top 3 line pairs per head, "
                        "layer-averaged):\n")
                for h in range(pair_heads.shape[0]):
                    h_pairs = [
                        (float(pair_heads[h, i, j]), i, j)
                        for i in range(K) for j in range(K) if i != j
                    ]
                    h_pairs.sort(reverse=True)
                    f.write(f"  head {h}:\n")
                    for val, i, j in h_pairs[:3]:
                        qi, kj = int(top_idx[i]), int(top_idx[j])
                        qz = int(atomic_numbers[qi]) if atomic_numbers is not None else -1
                        kz = int(atomic_numbers[kj]) if atomic_numbers is not None else -1
                        qs = _z_to_symbol(qz) if qz >= 0 else "?"
                        ks = _z_to_symbol(kz) if kz >= 0 else "?"
                        f.write(f"    {qs:>3s} {float(wavelength[qi]):8.2f} nm -> "
                                f"{ks:>3s} {float(wavelength[kj]):8.2f} nm   "
                                f"attn={val:.4e}\n")

            elem_mean = pair_data.get("elem_mean")
            elem_symbols = pair_data.get("elem_symbols", [])
            if elem_mean is not None and len(elem_symbols) > 0:
                E = len(elem_symbols)
                # Strongest off-diagonal element couplings.
                elem_pairs = [
                    (float(elem_mean[a, b]), a, b)
                    for a in range(E) for b in range(E) if a != b
                ]
                elem_pairs.sort(reverse=True)
                f.write("\nTop 10 element-to-element couplings "
                        "(mean attention per query line, off-diagonal):\n")
                for rank, (val, a, b) in enumerate(elem_pairs[:10]):
                    f.write(f"  {rank + 1:2d}. {elem_symbols[a]:>3s} -> "
                            f"{elem_symbols[b]:>3s}   attn={val:.4e}\n")
                diag = np.array([elem_mean[a, a] for a in range(E)])
                top_self = np.argsort(diag)[::-1][:5]
                f.write("\nStrongest within-element attention (diagonal):\n")
                for a in top_self:
                    f.write(f"  {elem_symbols[a]:>3s} -> {elem_symbols[a]:>3s}   "
                            f"attn={diag[a]:.4e}\n")

            f.write("\nHow to read these relationships:\n")
            f.write("- SAME-element pairs / large diagonal entries: the model\n")
            f.write("  cross-references several emission lines of one species. This is\n")
            f.write("  the expected physics-consistent behaviour (multiple lines of an\n")
            f.write("  element carry redundant concentration information, making the\n")
            f.write("  representation robust to noise or distortion of a single line,\n")
            f.write("  e.g. self-absorption of strong resonance lines).\n")
            f.write("- CROSS-element pairs: either (a) learned composition correlations\n")
            f.write("  in the sample matrix (elements that co-occur, e.g. alloying\n")
            f.write("  partners), or (b) spectral proximity / interference - check the\n")
            f.write("  wavelengths: pairs within a few nm likely disambiguate\n")
            f.write("  overlapping lines rather than encode chemistry.\n")
            f.write("- Values are raw softmax weights: each query row sums to 1 over\n")
            f.write("  ALL sequence positions, so sub-matrix rows do not sum to 1 and\n")
            f.write("  magnitudes shrink as the number of lines grows.\n")

        f.write("\n" + "=" * 50 + "\n")
        f.write("Caveats:\n")
        f.write("- Attention reflects the SHARED CLS representation, not a\n")
        f.write("  per-output-element attribution (the binned head's per-element\n")
        f.write("  branches all read the same pooled vector). Per-target-element\n")
        f.write("  attribution would require a gradient-based method.\n")
        f.write("- Per-channel importance is attention-informed but relies on the\n")
        f.write("  linear embedding projection; it is only produced for\n")
        f.write("  line_token_linear runs. intensity (bin) runs get per-bin only.\n")
    print(f"Saved: {path}")


def main(args):
    run_mgr = RunManager.from_existing_run(args.run_dir)
    config_path = run_mgr.run_dir / "config.yaml"
    run_info_path = run_mgr.run_dir / "run_info.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"config.yaml not found in {run_mgr.run_dir}")
    config = load_config(str(config_path))
    run_info = load_config(str(run_info_path)) if run_info_path.is_file() else {}

    checkpoint_path = run_mgr.get_checkpoint_for_mode("finetune")
    if checkpoint_path is None:
        raise ValueError(f"No checkpoint found in {run_mgr.checkpoint_dir}")
    print(f"Run: {run_mgr.run_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Embedding: {run_info.get('embedding_type', config['model'].get('embedding_type'))}")

    dataset, token_meta, source = prepare_data(config, run_info, args)
    print(f"Data source: {source}")

    encoder = build_encoder(config, run_info, token_meta)
    encoder_state = _checkpoint_encoder_state(str(checkpoint_path))
    _load_weights_shape_safe(encoder, encoder_state)

    ts = datetime.now().strftime("attention_importance_%Y-%m-%d_%H-%M-%S")
    output_dir = run_mgr.get_evaluation_dir(ts)
    print(f"Output: {output_dir}")

    run_analysis(encoder, dataset, token_meta, config, run_info, args, output_dir)

    print("\n" + "=" * 60)
    print(f"Attention importance analysis complete: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Attention-based input importance for fine-tuned LIBS models",
    )
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Fine-tuned run directory (contains checkpoints/, config.yaml)")
    parser.add_argument("--tokens_path", type=str, default=None,
                        help="Analyze spectra straight from this line_tokens_*.h5 cache "
                             "(lightweight; skips dataset rebuild and the test split).")
    parser.add_argument("--use_token_cache", action="store_true",
                        help="Use run_info.line_tokens_path directly instead of rebuilding "
                             "the labeled dataset / test split.")
    parser.add_argument("--max_samples", type=int, default=256,
                        help="Number of test spectra to aggregate attention over.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for the attention pass (keep small; attention "
                             "matrices are O(seq_len^2)).")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pair_top_k", type=int, default=40,
                        help="Number of top lines (by CLS importance) for the "
                             "line-line mutual attention matrix.")
    parser.add_argument("--pair_samples", type=int, default=64,
                        help="Number of spectra to aggregate for the mutual "
                             "(line-line) attention pass.")
    parser.add_argument("--skip_pairs", action="store_true",
                        help="Skip the mutual line-line attention analysis.")

    args = parser.parse_args()
    main(args)
