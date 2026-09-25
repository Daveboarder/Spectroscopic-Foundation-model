"""
Fine-tuning script for LIBS Foundation Model.

Loads a pre-trained model from a run directory, fine-tunes on labeled data,
saves checkpoints and run metadata.

Usage:
    uv run python train_finetune.py --pretrain_run_dir runs/pretrain_... --task both
"""

import argparse
import re
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
from data.dataset import LabeledLIBSDataset
from data.libs_pipeline import (
    build_dataset_from_config,
    cluster_compositions,
    extract_finetune_labels,
    get_or_make_splits,
)
from data.line_embedding_pipeline import (
    prepare_line_token_assets,
    prepare_line_tokens_assets,
)
from models.libs_transformer import LIBSTransformer
from training.finetune import LIBSFinetuneModule, FinetuneDataModule
from utils.run_manager import RunManager

# Plasma-state targets and grouped splits (contracts C2/C3 in data/libs_pipeline.py).
# Imported lazily-with-fallback so this script stays importable while the
# generator module is still being landed; the fallbacks give "no plasma labels"
# and the plain unique_id parser respectively.
try:
    from data.libs_pipeline import extract_plasma_targets
except ImportError:  # pragma: no cover - transitional
    extract_plasma_targets = None
try:
    from data.libs_pipeline import measured_groups
except ImportError:  # pragma: no cover - transitional
    measured_groups = None

SPLIT_STRATEGIES = ('random', 'group_sample', 'group_instrument')
PLASMA_TARGET_KEYS = ('Te', 'log10_Ne', 'log10_Nl', 'is_two_zone', 'has_plasma_labels')


def _fallback_plasma_targets(sample_table) -> dict[str, np.ndarray]:
    """C2-shaped zeros for tables without zone columns (measured data or a
    libs_pipeline that predates extract_plasma_targets)."""
    n = len(sample_table)
    return {k: np.zeros(n, dtype=np.float32) for k in PLASMA_TARGET_KEYS}


def plasma_targets_for_table(sample_table) -> dict[str, np.ndarray]:
    """`extract_plasma_targets` (C2) with the zeros fallback; every value is a
    float32 array of length len(sample_table)."""
    if extract_plasma_targets is None:
        return _fallback_plasma_targets(sample_table)
    aux = extract_plasma_targets(sample_table)
    out = {}
    for key in PLASMA_TARGET_KEYS:
        v = aux.get(key)
        out[key] = (np.zeros(len(sample_table), dtype=np.float32) if v is None
                    else np.asarray(v, dtype=np.float32).reshape(-1))
    return out


_RUN_SUFFIX_RE = re.compile(r"_R\d+$")


def _fallback_measured_groups(sample_table, by: str = "sample") -> np.ndarray:
    """Group ids from `unique_id` = `{sample}_{INSTRUMENT}_R{run:02d}` (the
    instrument token is `NAME_SERIAL`, e.g. REMUS_9951601): strip the
    `sample_type_id` prefix and the `_R<run>` suffix. Used only when
    data.libs_pipeline.measured_groups is unavailable."""
    if by == "sample":
        return sample_table["sample_type_id"].astype(str).to_numpy()
    if by != "instrument":
        raise ValueError(f"measured_groups: by must be 'sample' or 'instrument', got {by!r}")
    ids = sample_table["unique_id"].astype(str).to_numpy()
    samples = sample_table["sample_type_id"].astype(str).to_numpy()
    out = []
    for uid, sid in zip(ids, samples):
        rest = uid[len(sid) + 1:] if uid.startswith(sid + "_") else uid
        out.append(_RUN_SUFFIX_RE.sub("", rest))
    return np.asarray(out, dtype=str)


def groups_for_strategy(sample_table, strategy: str) -> np.ndarray | None:
    """Group ids for a split strategy (None for 'random')."""
    if strategy == 'random':
        return None
    if strategy not in SPLIT_STRATEGIES:
        raise ValueError(f"split strategy must be one of {SPLIT_STRATEGIES}, got {strategy!r}")
    by = 'sample' if strategy == 'group_sample' else 'instrument'
    fn = measured_groups if measured_groups is not None else _fallback_measured_groups
    return np.asarray(fn(sample_table, by=by)).astype(str)


class SaveRawEncoderCallback(Callback):
    """Save raw encoder weights every N epochs for easy mid-training evaluation."""
    def __init__(self, save_path: str, save_every_n_epochs: int = 1):
        self.save_path = save_path
        self.save_every_n_epochs = save_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.save_every_n_epochs == 0:
            torch.save(pl_module.encoder.state_dict(), self.save_path)
            info_path = str(self.save_path).replace('.pt', '_info.txt')
            with open(info_path, 'w') as f:
                f.write(f"epoch: {trainer.current_epoch + 1}\n")
                f.write(f"global_step: {trainer.global_step}\n")


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _checkpoint_state_dict(checkpoint) -> dict:
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        if any(k.startswith('model.') for k in state_dict.keys()):
            return {
                k[len('model.'):]: v for k, v in state_dict.items()
                if k.startswith('model.')
            }
        return state_dict
    return checkpoint


def _load_weights_shape_safe(model: LIBSTransformer, state_dict: dict) -> None:
    """Load only tensors with matching shapes (skips mip_head etc. when unused in finetune)."""
    model_sd = model.state_dict()
    filtered = {}
    skipped = []
    for key, value in state_dict.items():
        if key not in model_sd:
            continue
        if model_sd[key].shape != value.shape:
            skipped.append(key)
            continue
        filtered[key] = value

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if skipped:
        print(f"  Skipped {len(skipped)} keys due to shape mismatch (expected for mip_head when finetuning):")
        for k in skipped[:8]:
            print(f"    - {k}: checkpoint {tuple(state_dict[k].shape)} vs model {tuple(model_sd[k].shape)}")
        if len(skipped) > 8:
            print(f"    ... and {len(skipped) - 8} more")
    if missing:
        n_emb = sum(1 for k in missing if k.startswith('embedding.'))
        n_enc = sum(1 for k in missing if k.startswith('encoder_blocks.'))
        print(f"  Missing keys: {len(missing)} (embedding={n_emb}, encoder={n_enc}, other={len(missing) - n_emb - n_enc})")
        if n_emb > 0 and n_enc > 0:
            raise RuntimeError(
                "Checkpoint architecture does not match the finetune model — "
                "embedding and encoder weights could not be loaded. "
                "Use the same embedding_type and model config as pre-training "
                "(see pretrain run config.yaml / run_info.yaml)."
            )
    if unexpected:
        print(f"  Unexpected keys in checkpoint (ignored): {len(unexpected)}")


def align_config_with_pretrain_run(config: dict, args, pretrain_run_dir: str) -> None:
    """Match model/embedding settings to the pretrain run so checkpoints load correctly."""
    run_dir = Path(pretrain_run_dir)
    run_info_path = run_dir / "run_info.yaml"
    pretrain_cfg_path = run_dir / "config.yaml"
    if not run_info_path.is_file():
        return

    run_info = yaml.safe_load(open(run_info_path))
    pretrain_emb = run_info.get("embedding_type")
    finetune_emb = config.get("model", {}).get("embedding_type", "intensity")

    if pretrain_emb and pretrain_emb != finetune_emb:
        print(
            f"\nAligning with pretrain run: embedding_type {finetune_emb!r} -> {pretrain_emb!r} "
            f"(from {run_info_path})"
        )

    if pretrain_cfg_path.is_file():
        pretrain_cfg = yaml.safe_load(open(pretrain_cfg_path))
        if "model" in pretrain_cfg:
            config["model"].update(pretrain_cfg["model"])

    lec = run_info.get("line_embedding_config")
    if lec:
        if not args.line_embedding_config:
            args.line_embedding_config = lec
            print(f"  Inherited --line_embedding_config {lec}")
        elif Path(lec).resolve() != Path(args.line_embedding_config).resolve():
            print(
                f"  WARNING: --line_embedding_config ({args.line_embedding_config}) "
                f"differs from pretrain ({lec})"
            )


def _generate_legacy_labeled(config: dict, seed: int):
    """Original toy generator with 5 classes and 5-dim concentrations."""
    print("Generating legacy labeled synthetic data...")
    generator = SyntheticLIBSGenerator(
        n_bins=config['data']['n_bins'],
        noise_sigma=config['data']['synthetic']['noise_sigma'],
        peak_width_range=tuple(config['data']['synthetic']['peak_width_range']),
        intensity_variation=config['data']['synthetic']['intensity_variation'],
        seed=seed,
    )
    total = config['data']['synthetic']['labeled_samples']
    spectra, labels, concentrations = generator.generate_dataset(n_samples=total, return_labels=True)
    t = int(0.7 * total); v = int(0.85 * total)
    return {
        'train': (spectra[:t], labels[:t], concentrations[:t]),
        'val':   (spectra[t:v], labels[t:v], concentrations[t:v]),
        'test':  (spectra[v:], labels[v:], concentrations[v:]),
        'element_names': None,
    }


def _generate_libs_pipeline_labeled(
    config: dict,
    libs_config_path: str,
    seed: int,
    split_strategy: str | None = None,
):
    """Realistic physics-based labeled data.

    Loads (or generates from cache) the full synthetic dataset, extracts
    per-element concentrations from sample_table, and partitions everything
    into train/val/test using the splits JSON keyed by the cache fingerprint.

    Returns the same dict shape as the legacy path but with concentrations of
    shape [N, n_elements] (60 elements by default) and labels coming from
    KMeans clustering on the concentration vectors. Also returns `'aux'`:
    the plasma-state targets of `extract_plasma_targets` (Te, log10_Ne,
    log10_Nl, is_two_zone, has_plasma_labels) split like the labels, and
    `'split_strategy'` (random | group_sample | group_instrument; the CLI
    override wins over `downstream.splits.strategy`).
    """
    print(f"Generating LIBS-pipeline labeled data from {libs_config_path}...")
    libs_cfg = yaml.safe_load(open(libs_config_path))
    libs_cfg.setdefault('generation', {})
    # Respect the libs_data.yaml seed unless caller overrides.
    libs_cfg['generation'].setdefault('seed', seed)
    downstream = libs_cfg.get('downstream', {})

    ds = build_dataset_from_config(libs_cfg)
    if len(ds) == 0:
        raise RuntimeError("LIBS pipeline produced no labeled data — check sample matrix / DB.")

    spectra = ds.spectra.astype(np.float32)
    elements = downstream.get('elements_to_predict')  # None = all 60
    concentrations, element_names, sample_type_ids = extract_finetune_labels(
        ds.sample_table, elements=elements,
    )
    aux_all = plasma_targets_for_table(ds.sample_table)
    n_labelled = int(np.sum(aux_all['has_plasma_labels'] > 0))
    print(f"Plasma-state labels: {n_labelled}/{len(ds)} spectra "
          f"(two-zone: {int(np.sum(aux_all['is_two_zone'] > 0))})")

    # Override n_bins from the actual wavelength array
    actual_n_bins = spectra.shape[1]
    if config['data']['n_bins'] != actual_n_bins:
        print(f"Overriding n_bins: {config['data']['n_bins']} -> {actual_n_bins}")
        config['data']['n_bins'] = actual_n_bins
        config['model']['max_seq_len'] = actual_n_bins + 1

    # Override n_classes (= cluster count), n_elements (= concentration vector dim),
    # and n_concentration_bins (per-element bin count for the binned task).
    # Classification labels: `downstream.class_label` = "cluster" (KMeans on the
    # concentration vectors, `n_clusters` classes; default) or "sample_type"
    # (one class per row of the sample matrix, e.g. one per mineral).
    class_label = str(downstream.get('class_label', 'cluster'))
    if class_label == 'sample_type':
        _, cluster_labels = np.unique(sample_type_ids, return_inverse=True)
        cluster_labels = cluster_labels.astype(np.int64)
        n_clusters = int(cluster_labels.max()) + 1
        print(f"Classification labels: sample_type ({n_clusters} classes)")
    elif class_label == 'cluster':
        n_clusters = downstream.get('n_clusters', 10)
        cluster_labels = cluster_compositions(concentrations, n_clusters=n_clusters, seed=seed)
    else:
        raise ValueError(f"downstream.class_label must be 'cluster' or 'sample_type', got {class_label!r}")
    config['data']['n_classes'] = n_clusters
    config['data']['n_elements'] = len(element_names)
    config['data']['n_concentration_bins'] = downstream.get('n_concentration_bins', 1000)

    # Shared deterministic split — cached alongside the spectra. Pretrain and
    # finetune will read the same JSON so test set is consistent across phases.
    # Grouped strategies (C3) keep whole samples / instruments on one side so
    # measured evaluation does not leak replicate shots between train and test.
    split_cfg = downstream.get('splits', {})
    strategy = str(split_strategy or split_cfg.get('strategy', 'random'))
    if strategy not in SPLIT_STRATEGIES:
        raise ValueError(f"split strategy must be one of {SPLIT_STRATEGIES}, got {strategy!r}")
    split_kwargs = dict(
        n=len(spectra),
        cache_dir=ds.cache_dir,
        cache_key=ds.cache_key,
        val_fraction=split_cfg.get('val_fraction', 0.15),
        test_fraction=split_cfg.get('test_fraction', 0.15),
        seed=split_cfg.get('seed', seed),
    )
    if strategy != 'random':
        groups = groups_for_strategy(ds.sample_table, strategy)
        print(f"Split strategy {strategy}: {len(np.unique(groups))} groups")
        try:
            splits, splits_path = get_or_make_splits(**split_kwargs, groups=groups, strategy=strategy)
        except TypeError as exc:
            raise RuntimeError(
                "data.libs_pipeline.get_or_make_splits does not accept groups/strategy yet "
                f"(needed for split strategy {strategy!r})"
            ) from exc
    else:
        splits, splits_path = get_or_make_splits(**split_kwargs)
    print(f"Splits: train={len(splits['train'])}, val={len(splits['val'])}, "
          f"test={len(splits['test'])}  (saved to {splits_path})")
    print(f"Elements: {len(element_names)}  Clusters: {n_clusters}  "
          f"Sample types: {len(np.unique(sample_type_ids))}")

    def pick(idx):
        return spectra[idx], cluster_labels[idx], concentrations[idx]

    def pick_aux(idx):
        return {k: v[idx] for k, v in aux_all.items()}

    return {
        'train': pick(splits['train']),
        'val':   pick(splits['val']),
        'test':  pick(splits['test']),
        'aux': {
            'train': pick_aux(splits['train']),
            'val':   pick_aux(splits['val']),
            'test':  pick_aux(splits['test']),
        },
        'element_names': element_names,
        'splits': splits,
        'split_strategy': strategy,
        'libs_dataset': ds,
    }


def generate_labeled_data(
    config: dict,
    seed: int = 42,
    libs_config_path: str | None = None,
    split_strategy: str | None = None,
):
    if libs_config_path:
        return _generate_libs_pipeline_labeled(config, libs_config_path, seed,
                                               split_strategy=split_strategy)
    return _generate_legacy_labeled(config, seed)


def build_lod_vector(element_names: list[str], lod_config_path: str) -> tuple[np.ndarray, dict]:
    """Build a per-element limit-of-detection vector aligned to `element_names`.

    Reads `config/element_lod.yaml` (keys: `default_lod`, `limits_of_detection`).
    Elements missing from the config fall back to `default_lod`.

    Returns:
        (lod_vector [n_elements] float32, lod_map {element: lod}).
    """
    cfg = yaml.safe_load(open(lod_config_path))
    default_lod = float(cfg.get("default_lod", 1.0e-4))
    table = cfg.get("limits_of_detection", {}) or {}
    lod_map = {}
    vec = np.empty(len(element_names), dtype=np.float32)
    n_default = 0
    for i, name in enumerate(element_names):
        if name in table:
            vec[i] = float(table[name])
        else:
            vec[i] = default_lod
            n_default += 1
        lod_map[name] = float(vec[i])
    print(f"Loaded LODs from {lod_config_path}: {len(element_names)} elements "
          f"({n_default} using default_lod={default_lod:g})")
    return vec, lod_map


def compute_detection_pos_weight(
    concentrations: np.ndarray,
    lod_vector: np.ndarray,
    clip: tuple[float, float] = (0.1, 100.0),
) -> np.ndarray:
    """BCE positive-class weight per element = n_absent / n_present (clipped).

    Counters the heavy class imbalance of trace elements that are present in
    only a small fraction of spectra. Elements with no positives get the upper
    clip value (the loss term is then effectively inert for them).
    """
    present = (concentrations >= lod_vector[None, :]).astype(np.float64)
    pos = present.sum(axis=0)
    neg = present.shape[0] - pos
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(pos > 0, neg / np.maximum(pos, 1.0), clip[1])
    return np.clip(w, clip[0], clip[1]).astype(np.float32)


def assert_gpu_available(accelerator: str) -> None:
    """Fail fast with a clear message when CUDA is requested but not allocatable.

    Catches the common case of a stale training process still holding the GPU
    (this machine runs in Exclusive Process compute mode, so only one client
    can allocate on the device at a time).
    """
    if accelerator != "gpu":
        return
    if not torch.cuda.is_available():
        raise RuntimeError(
            "config device.accelerator is 'gpu' but torch.cuda.is_available() "
            "is False. Set device.accelerator: cpu in the config, or fix CUDA."
        )
    try:
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
    except RuntimeError as exc:
        raise RuntimeError(
            "CUDA GPU is busy or unavailable — another process is likely "
            "holding the device.\n"
            "  Check:  nvidia-smi\n"
            "  Stop:   kill <pid>   (or kill -9 <pid> if the process is stuck)\n"
            "  Then re-run this command."
        ) from exc


def load_pretrained_model(
    config: dict,
    checkpoint_path: str,
    line_dict_meta: dict | None = None,
    line_token_meta: dict | None = None,
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
    if emb_type == 'line_token' and line_dict_meta:
        kwargs['n_lines'] = line_dict_meta['n_lines']
        kwargs['line_dict_meta'] = line_dict_meta
        kwargs['n_elements_vocab'] = line_dict_meta.get('n_elements', 53)
    if emb_type == 'line_token_linear' and line_token_meta:
        kwargs['n_lines'] = line_token_meta['n_lines']
        kwargs['line_token_meta'] = line_token_meta
        kwargs['n_mip_target_channels'] = int(
            config['model'].get('n_mip_target_channels', 2)
        )
    model = LIBSTransformer(**kwargs)

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = _checkpoint_state_dict(checkpoint)
    _load_weights_shape_safe(model, state_dict)
    print(f"Loaded pre-trained weights from {checkpoint_path}")
    return model


def create_fresh_model(
    config: dict,
    line_dict_meta: dict | None = None,
    line_token_meta: dict | None = None,
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
    if emb_type == 'line_token' and line_dict_meta:
        kwargs['n_lines'] = line_dict_meta['n_lines']
        kwargs['line_dict_meta'] = line_dict_meta
        kwargs['n_elements_vocab'] = line_dict_meta.get('n_elements', 53)
    if emb_type == 'line_token_linear' and line_token_meta:
        kwargs['n_lines'] = line_token_meta['n_lines']
        kwargs['line_token_meta'] = line_token_meta
        kwargs['n_mip_target_channels'] = int(
            config['model'].get('n_mip_target_channels', 2)
        )
    model = LIBSTransformer(**kwargs)
    print(f"Created fresh model with {model.num_parameters:,} parameters")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# cf_quantification helpers (also reused by scripts/evaluate_cf.py)
# ─────────────────────────────────────────────────────────────────────────────
def finetune_checkpoint_path(run_dir: str | Path) -> Path:
    """`best.ckpt` of a fine-tune run when present, else the RunManager order."""
    run_dir = Path(run_dir)
    best = run_dir / "checkpoints" / "best.ckpt"
    if best.is_file():
        return best
    ckpt = RunManager.from_existing_run(str(run_dir)).get_checkpoint_for_mode("finetune")
    if ckpt is None or Path(ckpt).suffix != ".ckpt":
        raise FileNotFoundError(f"no Lightning checkpoint (best.ckpt / last.ckpt) in {run_dir}")
    return Path(ckpt)


def load_module_state_shape_safe(module: torch.nn.Module, checkpoint_path: str | Path) -> None:
    """Load a Lightning checkpoint's state_dict into `module`, keeping only
    tensors whose key and shape match (mirrors publication/inference_runner)."""
    ckpt_obj = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    sd = ckpt_obj["state_dict"] if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj else ckpt_obj
    mod_sd = module.state_dict()
    filtered = {k: v for k, v in sd.items() if k in mod_sd and mod_sd[k].shape == v.shape}
    missing, unexpected = module.load_state_dict(filtered, strict=False)
    n_skipped = len(sd) - len(filtered)
    if n_skipped or missing:
        print(f"  state_dict: loaded {len(filtered)}/{len(mod_sd)} tensors "
              f"(skipped {n_skipped} from checkpoint, {len(missing)} missing in checkpoint)")


def lod_vector_from_run_info(run_info: dict, element_names: list[str]) -> torch.Tensor:
    lod_map = run_info.get("element_lod") or {}
    default = float(run_info.get("default_lod", 1e-4))
    return torch.tensor([float(lod_map.get(n, default)) for n in element_names], dtype=torch.float32)


def _token_cache_line_dict_hash(tokens_path: str | Path | None) -> str | None:
    if not tokens_path or not Path(tokens_path).is_file():
        return None
    import h5py
    with h5py.File(tokens_path, "r") as f:
        h = f.attrs.get("line_dict_hash")
    return str(h) if h is not None else None


def align_model_section_with_pretrain(cfg: dict, pretrain_run: str | None) -> None:
    """Overlay `cfg['model']` with the pretrain run's model section (if any),
    exactly as align_config_with_pretrain_run does at training time."""
    if not pretrain_run:
        return
    pre_cfg_path = Path(pretrain_run) / "config.yaml"
    if not pre_cfg_path.is_file():
        return
    pre_cfg = yaml.safe_load(open(pre_cfg_path)) or {}
    if "model" in pre_cfg:
        cfg.setdefault("model", {}).update(pre_cfg["model"])


def load_seed_module(
    run_dir: str | Path,
    config: dict,
    token_meta: dict,
    line_dict_meta: dict | None = None,
    strict_tokens: bool = True,
    expected_element_names: list[str] | None = None,
) -> LIBSFinetuneModule:
    """Load a frozen fine-tuned seed (quantification_binned / detection) for the
    CF task, mirroring publication.inference_runner.FinetuneInferenceRunner.module.

    Args:
        run_dir: fine-tune run directory of the seed
        config: current (aligned) training config — fallback for `data.n_classes`
                and the model section when the seed run has no config.yaml
        token_meta: current token cache meta (prepare_line_tokens_assets) —
                    defines n_lines / feature normalisation of the encoder
        line_dict_meta: unused for line_token_linear seeds (kept for symmetry
                    with load_pretrained_model)
        strict_tokens: require the seed's `line_tokens_path` basename to equal
                    the current token cache basename (training). Evaluation on
                    another spectra cache (scripts/evaluate_cf.py) passes False
                    and only the line-dictionary hash / n_lines are compared.
        expected_element_names: raise if the seed's element order differs

    Returns:
        LIBSFinetuneModule in eval mode with all parameters frozen.
    """
    from analyze_attention_importance import _checkpoint_encoder_state, build_encoder

    run_dir = Path(run_dir)
    run_info_path = run_dir / "run_info.yaml"
    if not run_info_path.is_file():
        raise FileNotFoundError(f"seed run has no run_info.yaml: {run_dir}")
    run_info = yaml.safe_load(open(run_info_path))
    cfg_path = run_dir / "config.yaml"
    cfg = yaml.safe_load(open(cfg_path)) if cfg_path.is_file() else {}
    cfg.setdefault("model", dict(config.get("model", {})))
    cfg.setdefault("data", {})
    cfg["data"].setdefault("n_classes", config.get("data", {}).get("n_classes", 10))
    # The effective model section of a fine-tune run is the one of its pretrain
    # run (align_config_with_pretrain_run); run_dir/config.yaml is the raw copy.
    align_model_section_with_pretrain(cfg, run_info.get("pretrain_run"))

    emb_type = run_info.get("embedding_type") or cfg["model"].get("embedding_type", "intensity")
    if emb_type != "line_token_linear":
        raise ValueError(
            f"seed run {run_dir.name} has embedding_type={emb_type!r}; the CF task "
            "needs line_token_linear seeds (same token cache as the CF encoder)."
        )
    cfg["model"]["embedding_type"] = emb_type

    # Token-layout consistency between the seed and the current cache.
    seed_tokens = run_info.get("line_tokens_path")
    cur_tokens = token_meta.get("line_tokens_path")
    if strict_tokens:
        if not seed_tokens or not cur_tokens or Path(seed_tokens).name != Path(cur_tokens).name:
            raise AssertionError(
                f"seed run {run_dir.name} was trained on token cache "
                f"{Path(seed_tokens).name if seed_tokens else None}, but the current cache is "
                f"{Path(cur_tokens).name if cur_tokens else None}. Re-train the seed on the "
                "current tokens (same libs_data / line_embedding configs)."
            )
    else:
        seed_hash = _token_cache_line_dict_hash(seed_tokens)
        cur_hash = _token_cache_line_dict_hash(cur_tokens)
        if seed_hash and cur_hash and seed_hash != cur_hash:
            raise AssertionError(
                f"seed run {run_dir.name} used line dictionary {seed_hash}, current cache "
                f"uses {cur_hash} — token layouts differ."
            )
    n_lines = int(token_meta["n_lines"])
    cfg["data"]["n_bins"] = n_lines
    cfg["model"]["max_seq_len"] = n_lines + 1

    encoder = build_encoder(cfg, run_info, token_meta)
    ckpt = finetune_checkpoint_path(run_dir)
    enc_state = _checkpoint_encoder_state(str(ckpt))
    enc_sd = encoder.state_dict()
    encoder.load_state_dict(
        {k: v for k, v in enc_state.items() if k in enc_sd and enc_sd[k].shape == v.shape},
        strict=False,
    )

    element_names = list(run_info["element_names"])
    if expected_element_names is not None and element_names != list(expected_element_names):
        raise ValueError(
            f"seed run {run_dir.name} predicts {len(element_names)} elements in a different "
            f"order than the current run ({len(expected_element_names)}); C0 / presence "
            "columns would be misaligned."
        )
    module = LIBSFinetuneModule(
        encoder=encoder,
        task=run_info["task"],
        n_classes=int(cfg["data"]["n_classes"]),
        n_elements=int(run_info["n_elements"]),
        n_concentration_bins=int(run_info.get("n_concentration_bins", 1000)),
        pool=run_info.get("pool", "cls"),
        element_names=element_names,
        lod=lod_vector_from_run_info(run_info, element_names),
    )
    load_module_state_shape_safe(module, ckpt)
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    print(f"Loaded seed {run_info['task']} module from {ckpt} "
          f"(best {run_info.get('best_metric')})")
    return module


def build_cf_assets(
    args,
    config: dict,
    element_names: list[str],
    libs_data_config: str,
    token_meta: dict,
    spectra_cache_path: str | None,
    split_strategy: str | None,
    strict_tokens: bool = True,
) -> dict:
    """Everything the cf_quantification module needs: CF tables, the `finetune.cf`
    config with CLI overrides, the frozen seeds and the run_info `cf` block (C7).

    `args` must carry seed_binned_run_dir, seed_detection_run_dir,
    cf_pure_physics, cf_c0_source and element_lod_config.
    """
    from cf.tables import build_cf_tables  # lazy: cf/ is only needed for this task

    if element_names is None:
        raise ValueError("cf_quantification requires the LIBS data pipeline (--libs_data_config).")
    libs_cfg = yaml.safe_load(open(libs_data_config))
    db_path = str(Path(libs_cfg["paths"]["db"]).expanduser().resolve())
    cf_tables = build_cf_tables(element_names, db_path, args.element_lod_config)

    cf_cfg = dict(config.get("finetune", {}).get("cf", {}) or {})
    cf_cfg["pure_physics"] = bool(args.cf_pure_physics)
    cf_cfg["c0_source"] = str(args.cf_c0_source)
    cf_cfg["line_dict_path"] = token_meta.get("line_dict_path")

    seed_binned = seed_detection = None
    if args.seed_binned_run_dir:
        seed_binned = load_seed_module(
            args.seed_binned_run_dir, config, token_meta,
            strict_tokens=strict_tokens, expected_element_names=element_names,
        )
        if seed_binned.task != "quantification_binned":
            raise ValueError(f"--seed_binned_run_dir must be a quantification_binned run "
                             f"(got {seed_binned.task})")
    elif cf_cfg["c0_source"] == "binned":
        print("WARNING: --cf_c0_source binned without --seed_binned_run_dir → uniform C0")
    if args.seed_detection_run_dir:
        seed_detection = load_seed_module(
            args.seed_detection_run_dir, config, token_meta,
            strict_tokens=strict_tokens, expected_element_names=element_names,
        )
        if seed_detection.task != "detection":
            raise ValueError(f"--seed_detection_run_dir must be a detection run "
                             f"(got {seed_detection.task})")

    run_info_cf = {
        "seed_binned_run": str(args.seed_binned_run_dir) if args.seed_binned_run_dir else None,
        "seed_detection_run": str(args.seed_detection_run_dir) if args.seed_detection_run_dir else None,
        "pure_physics": bool(args.cf_pure_physics),
        "cf_cfg": {k: (v if isinstance(v, (int, float, str, bool, type(None), list, dict)) else str(v))
                   for k, v in cf_cfg.items()},
        "line_dict_path": cf_cfg["line_dict_path"],
        "spectra_cache_path": spectra_cache_path,
        "split_strategy": split_strategy,
        "c0_source": cf_cfg["c0_source"],
    }
    return {
        "cf_tables": cf_tables,
        "cf_cfg": cf_cfg,
        "seed_binned": seed_binned,
        "seed_detection": seed_detection,
        "run_info_cf": run_info_cf,
        "lod_map": {n: float(cf_tables.lod[i]) for i, n in enumerate(element_names)},
    }


def main(args):
    config = load_config(args.config)

    # Resolve pretrained checkpoint
    pretrained_checkpoint = None
    pretrain_run_dir = None

    if args.pretrain_run_dir:
        pretrain_mgr = RunManager.from_existing_run(args.pretrain_run_dir)
        pretrain_run_dir = str(pretrain_mgr.run_dir)
        align_config_with_pretrain_run(config, args, pretrain_run_dir)
        pretrained_checkpoint = pretrain_mgr.get_checkpoint_for_mode("pretrain")
        if pretrained_checkpoint is None:
            raise ValueError(f"No checkpoint found in pretrain run: {args.pretrain_run_dir}")
        print(f"Using pretrained model from: {pretrain_run_dir}")
        print(f"  checkpoint: {pretrained_checkpoint}")

    # Create run manager
    experiment_name = args.experiment_name or args.task
    run_mgr = RunManager(
        run_type="finetune",
        experiment_name=experiment_name,
        base_dir=args.runs_dir,
        config_path=args.config,
    )

    pl.seed_everything(args.seed)

    line_dict_meta = None
    line_token_meta = None
    line_features_path = None
    line_tokens_path = None
    train_indices = val_indices = test_indices = None

    requested_emb = config['model'].get('embedding_type', 'intensity')
    if args.line_embedding_config and requested_emb == 'intensity':
        requested_emb = 'line_token_linear'
        config['model']['embedding_type'] = requested_emb
    use_line_token = requested_emb in ('line_token', 'line_token_linear')
    if use_line_token:
        if not args.line_embedding_config or not args.libs_data_config:
            raise ValueError(
                f"embedding_type={requested_emb!r} finetune requires "
                "--line_embedding_config and --libs_data_config"
            )

    data = generate_labeled_data(
        config, seed=args.seed, libs_config_path=args.libs_data_config,
        split_strategy=getattr(args, 'split_strategy', None),
    )
    train_spectra, train_labels, train_conc = data['train']
    val_spectra, val_labels, val_conc = data['val']
    test_spectra, test_labels, test_conc = data['test']
    element_names = data.get('element_names')
    aux = data.get('aux') or {}
    train_aux, val_aux, test_aux = aux.get('train'), aux.get('val'), aux.get('test')
    split_strategy = data.get('split_strategy')
    spectra_cache_path = None
    if 'libs_dataset' in data and hasattr(data['libs_dataset'], '_cache_path'):
        spectra_cache_path = str(data['libs_dataset']._cache_path())

    if use_line_token and 'libs_dataset' in data:
        ds = data['libs_dataset']
        if requested_emb == 'line_token':
            line_dict_meta = prepare_line_token_assets(
                ds.spectra.astype(np.float32),
                ds.wavelength,
                args.line_embedding_config,
                spectra_cache_key=ds.cache_key,
                verbose=True,
            )
            n_lines = line_dict_meta['n_lines']
            line_features_path = line_dict_meta['line_features_path']
        else:
            line_token_meta = prepare_line_tokens_assets(
                ds.spectra.astype(np.float32),
                ds.wavelength,
                args.line_embedding_config,
                spectra_cache_key=ds.cache_key,
                verbose=True,
            )
            n_lines = line_token_meta['n_lines']
            line_tokens_path = line_token_meta['line_tokens_path']
        config['data']['n_bins'] = n_lines
        config['model']['max_seq_len'] = n_lines + 1
        splits = data['splits']
        train_indices = splits['train']
        val_indices = splits['val']
        test_indices = splits['test']
        print(f"Line tokens: {n_lines} lines per spectrum")

    print(f"Train: {len(train_labels)}, Val: {len(val_labels)}, Test: {len(test_labels)}")
    if element_names is not None:
        print(f"Predicting {len(element_names)} elements: {element_names[:10]}"
              f"{'...' if len(element_names) > 10 else ''}")

    if pretrained_checkpoint:
        encoder = load_pretrained_model(
            config, str(pretrained_checkpoint),
            line_dict_meta=line_dict_meta,
            line_token_meta=line_token_meta,
        )
    else:
        print("No pre-trained checkpoint provided, creating fresh model...")
        encoder = create_fresh_model(
            config, line_dict_meta=line_dict_meta, line_token_meta=line_token_meta,
        )

    # n_elements: prefer config.data.n_elements (set by the libs pipeline path);
    # fall back to n_classes for the legacy 5-class generator.
    n_elements = config['data'].get('n_elements', config['data']['n_classes'])
    n_concentration_bins = config['data'].get('n_concentration_bins', 1000)

    ft_epochs = int(config['finetune']['epochs'])
    ft_warmup = int(config['finetune'].get('warmup_epochs', min(1, max(0, ft_epochs - 1))))
    ft_warmup = max(1, min(ft_warmup, ft_epochs - 1)) if ft_epochs > 1 else 1

    # Detection task: derive per-element LOD thresholds + class-imbalance weights.
    lod_vector = None
    lod_pos_weight = None
    lod_map = None
    # Detection needs the LOD as its label threshold; the concentration tasks
    # only use it for the per-element log_rmse / within_2x block, so it is built
    # whenever element names are known and never required there.
    if element_names is not None and args.task in (
            'detection', 'quantification_binned', 'quantification', 'regression'):
        lod_np, lod_map = build_lod_vector(element_names, args.element_lod_config)
        lod_vector = torch.from_numpy(lod_np)
    if args.task == 'detection':
        if element_names is None:
            raise ValueError(
                "detection task requires the LIBS data pipeline "
                "(--libs_data_config) so element names are known."
            )
        pw_np = compute_detection_pos_weight(train_conc, lod_np)
        lod_pos_weight = torch.from_numpy(pw_np)
        present_frac = (train_conc >= lod_np[None, :]).mean(axis=0)
        print("Detection labels (train present fraction per element):")
        for name, frac in sorted(zip(element_names, present_frac),
                                 key=lambda kv: kv[1], reverse=True):
            print(f"  {name:>3s}: present={frac:6.2%}  lod={lod_map[name]:.1e}")

    # CF task: physics tables, frozen seeds and the `finetune.cf` block.
    cf_assets = None
    cf_module_kwargs: dict = {}
    if args.task == 'cf_quantification':
        if line_token_meta is None:
            raise ValueError(
                "cf_quantification requires the line_token_linear path "
                "(--libs_data_config + --line_embedding_config)."
            )
        cf_assets = build_cf_assets(
            args, config, element_names, args.libs_data_config, line_token_meta,
            spectra_cache_path=spectra_cache_path, split_strategy=split_strategy,
        )
        cf_module_kwargs = {
            'cf_tables': cf_assets['cf_tables'],
            'cf_cfg': cf_assets['cf_cfg'],
            'seed_binned': cf_assets['seed_binned'],
            'seed_detection': cf_assets['seed_detection'],
        }
        lod_map = cf_assets['lod_map']
        n_labelled_train = int(np.sum(train_aux['has_plasma_labels'] > 0)) if train_aux else 0
        print(f"CF task: pure_physics={args.cf_pure_physics}, c0_source={args.cf_c0_source}, "
              f"presence_gate={cf_assets['cf_cfg'].get('presence_gate', True)}, "
              f"train shots with plasma labels: {n_labelled_train}/{len(train_labels)}")
        if n_labelled_train == 0:
            print("WARNING: no training shot carries plasma labels — the CF heads will not "
                  "receive any gradient (measured data never trains the CF task).")

    finetune_module = LIBSFinetuneModule(
        encoder=encoder,
        task=args.task,
        n_classes=config['data']['n_classes'],
        n_elements=n_elements,
        n_concentration_bins=n_concentration_bins,
        freeze_encoder=args.freeze_encoder,
        learning_rate=config['finetune']['learning_rate'],
        weight_decay=config['finetune']['weight_decay'],
        warmup_epochs=ft_warmup,
        max_epochs=ft_epochs,
        pool=args.pool,
        element_names=element_names,
        lod=lod_vector,
        detection_pos_weight=lod_pos_weight,
        **cf_module_kwargs,
    )

    needs_concentrations = args.task in ('regression', 'quantification', 'quantification_binned',
                                         'detection', 'cf_quantification', 'both')
    needs_aux = args.task == 'cf_quantification'
    spectra_unused = line_features_path is not None or line_tokens_path is not None
    data_module = FinetuneDataModule(
        train_spectra=train_spectra if not spectra_unused else None,
        train_labels=train_labels,
        val_spectra=val_spectra if not spectra_unused else None,
        val_labels=val_labels,
        train_concentrations=train_conc if needs_concentrations else None,
        val_concentrations=val_conc if needs_concentrations else None,
        batch_size=config['finetune']['batch_size'],
        num_workers=args.num_workers,
        line_features_path=line_features_path,
        line_tokens_path=line_tokens_path,
        train_indices=train_indices,
        val_indices=val_indices,
        train_aux=train_aux if needs_aux else None,
        val_aux=val_aux if needs_aux else None,
    )

    # Logger
    logger_type = config['logging'].get('logger', 'tensorboard')

    if logger_type == 'wandb':
        wandb_config = config['logging'].get('wandb', {})
        logger = WandbLogger(
            project=wandb_config.get('project', 'libs-foundation-model'),
            entity=wandb_config.get('entity'),
            name=run_mgr.run_name,
            tags=wandb_config.get('tags', []) + ['finetune', args.task],
            save_dir=str(run_mgr.log_dir),
            config={
                'model': config['model'],
                'finetune': config['finetune'],
                'task': args.task,
                'pretrain_run': pretrain_run_dir,
                'run_dir': str(run_mgr.run_dir),
            },
        )
    else:
        logger = TensorBoardLogger(
            save_dir=str(run_mgr.log_dir),
            name='',
            version='',
        )

    # Monitor metric — task-specific so the best checkpoint reflects the right goal
    if args.task == 'classification':
        monitor, mon_mode = 'val/accuracy', 'max'
    elif args.task in ('regression', 'quantification'):
        monitor, mon_mode = 'val/reg_mae', 'min'
    elif args.task == 'quantification_binned':
        monitor, mon_mode = 'val/bin_accuracy', 'max'
    elif args.task == 'detection':
        monitor, mon_mode = 'val/det_f1', 'max'
    elif args.task == 'cf_quantification':
        monitor, mon_mode = 'val/cf_log_rmse', 'min'
    else:
        monitor, mon_mode = 'val/loss', 'min'

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(run_mgr.checkpoint_dir),
        filename='best',
        save_top_k=1,
        monitor=monitor,
        mode=mon_mode,
        save_last=True,
        auto_insert_metric_name=False,
    )

    raw_encoder_callback = SaveRawEncoderCallback(
        save_path=str(run_mgr.checkpoint_dir / 'encoder_latest.pt'),
        save_every_n_epochs=1,
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    callbacks = [checkpoint_callback, lr_monitor, raw_encoder_callback]
    if args.early_stopping:
        callbacks.append(EarlyStopping(
            monitor=monitor, patience=10, mode=mon_mode, verbose=True,
        ))

    # Run info
    run_info_common = {
        "task": args.task,
        "pool": args.pool,
        "embedding_type": config['model'].get('embedding_type', 'intensity'),
        "n_elements": n_elements,
        "n_concentration_bins": n_concentration_bins,
        "element_names": element_names,
        "libs_data_config": args.libs_data_config,
        "line_embedding_config": args.line_embedding_config,
        "line_features_path": line_features_path,
        "line_tokens_path": line_tokens_path,
        "spectra_cache_path": spectra_cache_path,
        "split_strategy": split_strategy,
        "model_params": encoder.num_parameters,
        "train_samples": len(train_labels),
        "val_samples": len(val_labels),
        "test_samples": len(test_labels),
        "freeze_encoder": args.freeze_encoder,
        "pretrain_run": pretrain_run_dir,
        "seed": args.seed,
    }
    if cf_assets is not None:
        run_info_common["cf"] = cf_assets["run_info_cf"]
    run_mgr.save_run_info({
        **run_info_common,
        "pretrain_checkpoint": str(pretrained_checkpoint) if pretrained_checkpoint else None,
        "status": "running",
    })

    # Trainer
    trainer = pl.Trainer(
        accelerator=config['device']['accelerator'],
        devices=1,
        precision=config['device']['precision'],
        max_epochs=config['finetune']['epochs'],
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=config['logging']['log_every_n_steps'],
        gradient_clip_val=1.0,
        deterministic=True,
    )

    print(f"\nStarting fine-tuning for task: {args.task}")
    print(f"Checkpoints: {run_mgr.checkpoint_dir}")
    print(f"Logs: {run_mgr.log_dir}")

    assert_gpu_available(config['device']['accelerator'])
    trainer.fit(finetune_module, data_module)

    # Test
    print("\nEvaluating on test set...")
    if line_tokens_path:
        from data.dataset import LineTokensLabeledDataset
        test_dataset = LineTokensLabeledDataset(
            line_tokens_path,
            test_labels,
            concentrations=test_conc if needs_concentrations else None,
            indices=test_indices,
            aux_targets=test_aux if needs_aux else None,
        )
    elif line_features_path:
        from data.dataset import LineTokenLabeledDataset
        test_dataset = LineTokenLabeledDataset(
            line_features_path,
            test_labels,
            concentrations=test_conc if needs_concentrations else None,
            indices=test_indices,
            aux_targets=test_aux if needs_aux else None,
        )
    else:
        test_dataset = LabeledLIBSDataset(
            spectra=test_spectra, labels=test_labels,
            concentrations=test_conc if needs_concentrations else None,
        )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config['finetune']['batch_size'],
        shuffle=False, num_workers=args.num_workers,
    )

    best_model = LIBSFinetuneModule.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        encoder=encoder, task=args.task,
        n_classes=config['data']['n_classes'],
        n_elements=n_elements,
        n_concentration_bins=n_concentration_bins,
        element_names=element_names,
        lod=lod_vector,
        detection_pos_weight=lod_pos_weight,
        **cf_module_kwargs,
    )
    test_results = trainer.test(best_model, dataloaders=test_loader)
    aggregate_test_results = dict(test_results[0]) if test_results else {}
    per_element_test = getattr(best_model, "test_per_element_metrics", {}) or {}
    if per_element_test:
        aggregate_test_results["per_element"] = per_element_test
    detection_test = getattr(best_model, "test_detection_metrics", {}) or {}
    if detection_test:
        aggregate_test_results["detection"] = detection_test
    plasma_test = getattr(best_model, "test_plasma_metrics", {}) or {}
    if plasma_test:
        aggregate_test_results["test_plasma_metrics"] = plasma_test

    # Save final raw encoder weights
    final_path = run_mgr.checkpoint_dir / 'final_encoder.pt'
    torch.save(encoder.state_dict(), final_path)
    print(f"\nSaved final encoder to {final_path}")

    run_mgr.save_run_info({
        **run_info_common,
        "status": "completed",
        "best_metric": float(checkpoint_callback.best_model_score) if checkpoint_callback.best_model_score else None,
        "final_encoder": str(final_path),
        "element_lod_config": (args.element_lod_config
                               if args.task in ('detection', 'cf_quantification') else None),
        "element_lod": lod_map,
        "test_results": aggregate_test_results if aggregate_test_results else None,
    })

    print("\n" + "="*60)
    print("Fine-tuning complete!")
    print("="*60)
    print(f"Run directory: {run_mgr.run_dir}")
    if checkpoint_callback.best_model_score:
        print(f"Best {monitor}: {checkpoint_callback.best_model_score:.4f}")
    if aggregate_test_results:
        print("\nTest metrics (aggregate):")
        for k in ("test/bin_accuracy", "test/decoded_mae", "test/decoded_r2",
                  "test/det_f1", "test/det_accuracy", "test/det_precision",
                  "test/det_recall", "test/cf_log_rmse", "test/cf_within2x",
                  "test/cf_r2_major", "test/te_mape", "test/ne_log_mae", "test/loss"):
            if k in aggregate_test_results:
                print(f"  {k}: {float(aggregate_test_results[k]):.6f}")
    if plasma_test:
        print("\nCF plasma-state recovery (one-zone synthetic test shots):")
        for k, v in plasma_test.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    if detection_test:
        print("\nDetection metrics:")
        for k in ("macro_f1", "micro_f1", "element_accuracy", "exact_match"):
            print(f"  {k}: {detection_test[k]:.4f}")
        print("\nPer-element detection (precision / recall / f1, support):")
        names = element_names if element_names is not None else sorted(detection_test["per_element"])
        for name in names:
            m = detection_test["per_element"].get(name)
            if m is None:
                continue
            print(f"  {name:>3s}: P={m['precision']:.3f} R={m['recall']:.3f} "
                  f"F1={m['f1']:.3f}  (support={int(m['support'])}, lod={m['lod']:.1e})")
    if per_element_test:
        print("\nTest metrics (per element):")
        names = element_names if element_names is not None else sorted(per_element_test.keys())
        for name in names:
            if name not in per_element_test:
                continue
            m = per_element_test[name]
            line = (f"  {name}: mae={m['mae']:.6f}, r2={m['r2']:.6f}, "
                    f"pearson={m['pearson']:.6f}, spearman={m['spearman']:.6f}")
            if 'log_rmse' in m:
                line += (f", log_rmse={m['log_rmse']:.4f}, within_2x={m['within_2x']:.3f}, "
                         f"n_censored={int(m['n_censored'])}")
            print(line)
    print(f"\nTo evaluate:")
    if args.task == 'cf_quantification':
        print(f"  uv run python scripts/evaluate_cf.py --run_dir {run_mgr.run_dir} "
              f"--libs_data_config config/libs_data_measured.yaml")
    else:
        print(f"  uv run python evaluate_model.py --run_dir {run_mgr.run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune LIBS Foundation Model")

    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--runs_dir', type=str, default='runs')
    parser.add_argument('--pretrain_run_dir', type=str, default=None,
                        help='Path to pretrain run directory')
    parser.add_argument('--task', type=str,
                        choices=['classification', 'quantification',
                                 'quantification_binned', 'detection',
                                 'cf_quantification',
                                 'regression', 'both'],
                        default='both',
                        help='Downstream task: classification (cluster ID), '
                             'quantification (MSE regression on concentrations), '
                             'quantification_binned (per-element bin CE, upstream-style), '
                             'detection (multi-label element presence/absence vs LOD), '
                             'cf_quantification (calibration-free Saha–Boltzmann solver '
                             'seeded by frozen binned/detection runs; needs the '
                             'line_token_linear path), '
                             'regression/both (legacy, kept for backward compat)')
    parser.add_argument('--element_lod_config', type=str,
                        default='config/element_lod.yaml',
                        help='Per-element limit-of-detection YAML used to derive '
                             'presence/absence labels for the detection task and the '
                             'censoring thresholds of cf_quantification.')
    # cf_quantification only
    parser.add_argument('--seed_binned_run_dir', type=str, default=None,
                        help='[cf] fine-tune run (task quantification_binned) on the SAME '
                             'token cache; its argmax concentrations seed the closure (C0).')
    parser.add_argument('--seed_detection_run_dir', type=str, default=None,
                        help='[cf] fine-tune run (task detection) on the SAME token cache; '
                             'presence >= 0.5 gates the per-line weights.')
    parser.add_argument('--cf_pure_physics', action='store_true',
                        help='[cf] zero-parameter variant: classical line weights '
                             '(cf.classical) and solver default init instead of the '
                             'learned heads.')
    parser.add_argument('--cf_c0_source', type=str, choices=['binned', 'uniform', 'truth'],
                        default='binned',
                        help='[cf] closure seed: binned (seed run argmax), uniform, or '
                             'truth (batch concentrations — debugging only).')
    parser.add_argument('--split_strategy', type=str, choices=list(SPLIT_STRATEGIES),
                        default=None,
                        help='Override downstream.splits.strategy of the libs data config: '
                             'random | group_sample | group_instrument (grouped strategies '
                             'keep whole measured samples / instruments on one side).')
    parser.add_argument('--libs_data_config', type=str, default=None,
                        help='Path to physics-based LIBS data pipeline config '
                             '(e.g. config/libs_data.yaml). If set, replaces the '
                             'legacy 5-class SyntheticLIBSGenerator.')
    parser.add_argument('--line_embedding_config', type=str, default=None,
                        help='Path to config/line_embedding.yaml for line-as-token mode.')
    parser.add_argument('--pool', type=str, choices=['cls', 'mean', 'cls_mean'],
                        default='cls',
                        help='How to pool encoder outputs for the heads: '
                             'cls (CLS token only), mean (mean over bins), '
                             'cls_mean (concat of both, head input is 2*d_model)')
    parser.add_argument('--freeze_encoder', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--num_workers', type=int, default=0)

    args = parser.parse_args()
    main(args)
