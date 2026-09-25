"""
Embedding-aware checkpoint inference for publication / comparison figures.

Supports intensity (raw wavelength bins) and line_token_linear runs on the
same test-split indices, for the ``detection`` task
(:meth:`FinetuneInferenceRunner.run_detection_inference`) and the
calibration-free ``cf_quantification`` task
(:meth:`FinetuneInferenceRunner.run_cf_inference`).

For a ``cf_quantification`` run the module is rebuilt with the CF tables, the
CF layer configuration and (when available) the frozen binned/detection seed
modules recorded under ``run_info['cf']`` (see contract C7 in the plan).
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
import yaml

from analyze_attention_importance import _checkpoint_encoder_state, build_encoder
from data.line_tokenization import FEATURE_NAMES
from models.heads import concentration_to_presence
from training.finetune import LIBSFinetuneModule
from utils.run_manager import RunManager

CF_TASK = "cf_quantification"
DEFAULT_DB_PATH = "external_data/Source/LIBS_data.db"
DEFAULT_LOD_CONFIG = "config/element_lod.yaml"


def resolve_cache_path(recorded: str | None, cache_dir: Path, pattern: str) -> Path:
    if recorded:
        p = Path(recorded)
        if p.is_file():
            return p
        local = cache_dir / p.name
        if local.is_file():
            return local
    candidates = sorted(cache_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"no cache file matching {pattern} in {cache_dir}")
    return candidates[0]


def _recorded_spectra_cache(run_info: dict | None) -> str | None:
    """`spectra_cache_path` recorded by a run (CF block first, then top level)."""
    if not run_info:
        return None
    cf_block = run_info.get("cf") or {}
    return cf_block.get("spectra_cache_path") or run_info.get("spectra_cache_path")


def resolve_spectra_cache(
    cache_dir: Path, n_total: int, run_info: dict | None = None,
) -> Path:
    """Locate the spectra cache (synthetic or measured) a run was trained on.

    Preference order:
      1. ``spectra_cache_path`` recorded in ``run_info`` (under ``cf`` or at the
         top level), resolved as an absolute path or by file name in
         ``cache_dir``;
      2. the first ``synthetic_cache_*.h5`` / ``measured_cache_*.h5`` whose
         spectrum count equals ``n_total``;
      3. the first ``synthetic_cache_*.h5`` (legacy behaviour).
    """
    recorded = _recorded_spectra_cache(run_info)
    if recorded:
        p = Path(recorded)
        for cand in (p, cache_dir / p.name):
            if cand.is_file():
                return cand
        print(f"[inference] recorded spectra cache {recorded} not found; "
              "matching by spectrum count instead")
    for pattern in ("synthetic_cache_*.h5", "measured_cache_*.h5"):
        for cand in sorted(cache_dir.glob(pattern)):
            with h5py.File(cand, "r") as f:
                if f["spectra"].shape[0] == n_total:
                    return cand
    return resolve_cache_path(None, cache_dir, "synthetic_cache_*.h5")


def load_splits(
    cache_dir: Path, n_total: int, n_test: int, strategy: str | None = None,
) -> dict[str, np.ndarray]:
    """Find the ``splits_*.json`` matching the run's sample counts.

    ``strategy`` (contract C3: ``random`` | ``group_sample`` |
    ``group_instrument``) prefers ``splits_<key>_<strategy>.json`` when several
    files match the counts; ``random`` splits keep the plain ``splits_<key>.json``
    name.
    """
    matches: list[Path] = []
    for cand in sorted(cache_dir.glob("splits_*.json")):
        s = json.load(open(cand))
        if len(s.get("test", [])) == n_test and sum(len(v) for v in s.values()) == n_total:
            matches.append(cand)
    if not matches:
        raise FileNotFoundError(
            f"no splits_*.json matching n_total={n_total}, n_test={n_test} in {cache_dir}",
        )
    chosen = matches[0]
    if strategy and strategy != "random":
        for cand in matches:
            if cand.stem.endswith(f"_{strategy}"):
                chosen = cand
                break
    s = json.load(open(chosen))
    return {k: np.asarray(v, dtype=np.int64) for k, v in s.items()}


def filter_indices_with_valid_tokens(tokens_path: Path, indices: np.ndarray) -> np.ndarray:
    """Keep only spectra that have at least one valid Voigt fit (token path)."""
    with h5py.File(tokens_path, "r") as f:
        valid = f["fit_valid"]
        keep = [int(i) for i in indices if valid[int(i)].sum() > 0]
    return np.asarray(keep, dtype=np.int64)


def read_sample_table(h5_path: str | Path) -> pd.DataFrame:
    """Read ``sample_table`` of a spectra cache as a DataFrame (no spectra)."""
    with h5py.File(h5_path, "r") as f:
        grp = f["sample_table"]
        if "columns" in grp.attrs:
            cols = json.loads(grp.attrs["columns"])
        else:
            cols = list(grp.keys())
        data = {}
        for c in cols:
            v = grp[c][:]
            if v.dtype.kind in ("S", "O"):
                v = [s.decode("utf-8") if isinstance(s, bytes) else s for s in v]
            data[c] = v
    return pd.DataFrame(data)


def _plasma_targets_fallback(sample_table: pd.DataFrame) -> dict[str, np.ndarray]:
    """Local implementation of contract C2 (``extract_plasma_targets``).

    Used only when ``data.libs_pipeline`` does not yet export the function.
    Returns zeros with ``has_plasma_labels = 0`` for measured data (columns
    missing or all zero).
    """
    n = len(sample_table)
    zeros = np.zeros(n, dtype=np.float32)
    out = {"Te": zeros.copy(), "log10_Ne": zeros.copy(), "log10_Nl": zeros.copy(),
           "is_two_zone": zeros.copy(), "has_plasma_labels": zeros.copy()}
    te_col = "Te1" if "Te1" in sample_table.columns else "Te"
    ne_col = "Ne1" if "Ne1" in sample_table.columns else "Ne"
    if te_col not in sample_table.columns or ne_col not in sample_table.columns:
        return out
    te = np.nan_to_num(sample_table[te_col].to_numpy(dtype=np.float64))
    ne = np.nan_to_num(sample_table[ne_col].to_numpy(dtype=np.float64))
    if not np.any(te > 0) or not np.any(ne > 0):
        return out
    has = ((te > 0) & (ne > 0)).astype(np.float32)
    out["Te"] = te.astype(np.float32)
    with np.errstate(divide="ignore"):
        out["log10_Ne"] = np.where(ne > 0, np.log10(np.maximum(ne, 1e-300)), 0.0).astype(np.float32)
        if "N1" in sample_table.columns and "l_inner" in sample_table.columns:
            nl = (np.nan_to_num(sample_table["N1"].to_numpy(dtype=np.float64))
                  * np.nan_to_num(sample_table["l_inner"].to_numpy(dtype=np.float64)))
            out["log10_Nl"] = np.where(nl > 0, np.log10(np.maximum(nl, 1e-300)), 0.0).astype(np.float32)
    if "plasma_model" in sample_table.columns:
        pm = sample_table["plasma_model"].astype(str).to_numpy()
        out["is_two_zone"] = (pm == "two_zone").astype(np.float32)
    out["has_plasma_labels"] = has
    return out


def plasma_targets_from_table(sample_table: pd.DataFrame) -> dict[str, np.ndarray]:
    """Contract C2 targets (Te, log10_Ne, log10_Nl, is_two_zone,
    has_plasma_labels) for every row of a spectra-cache sample table.

    Delegates to ``data.libs_pipeline.extract_plasma_targets`` when present and
    falls back to a local implementation of the same contract otherwise.
    """
    try:
        from data.libs_pipeline import extract_plasma_targets  # type: ignore
    except ImportError:
        extract_plasma_targets = None
    if extract_plasma_targets is not None:
        try:
            out = extract_plasma_targets(sample_table)
            return {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}
        except Exception as exc:  # pragma: no cover - defensive against API drift
            print(f"[inference] extract_plasma_targets failed ({exc}); using fallback")
    return _plasma_targets_fallback(sample_table)


def _to_numpy(t: Any) -> np.ndarray | None:
    if t is None:
        return None
    if isinstance(t, torch.Tensor):
        t = t.detach()
        if t.dtype == torch.bool:
            return t.cpu().numpy()
        return t.float().cpu().numpy()
    return np.asarray(t)


class FinetuneInferenceRunner:
    """Run detection / CF inference for one fine-tuned run."""

    def __init__(
        self,
        run_dir: str | Path,
        device: str = "auto",
        batch_size: int = 32,
        label: str = "",
    ):
        self.run_dir = Path(run_dir)
        self.label = label or self.run_dir.name
        self.batch_size = batch_size
        self.device = (
            "cuda" if device == "auto" and torch.cuda.is_available()
            else ("cpu" if device == "auto" else device)
        )
        self.config = yaml.safe_load(open(self.run_dir / "config.yaml"))
        self.run_info = yaml.safe_load(open(self.run_dir / "run_info.yaml"))
        self.element_names: list[str] = list(self.run_info["element_names"])
        self.task: str = str(self.run_info.get("task", "quantification_binned"))
        self.cf_info: dict = dict(self.run_info.get("cf") or {})
        self.embedding_type = str(
            self.run_info.get("embedding_type")
            or self.config.get("model", {}).get("embedding_type", "intensity"),
        )
        self.cache_dir = Path("external_data/cache")
        self._module: LIBSFinetuneModule | None = None
        self._token_meta: dict | None = None
        self._spectra_path: Path | None = None
        self._tokens_path: Path | None = None
        self._concentrations: np.ndarray | None = None
        self._lod: torch.Tensor | None = None
        self._cf_tables = None
        self._sample_table: pd.DataFrame | None = None
        self._plasma_targets: dict[str, np.ndarray] | None = None

    @property
    def is_cf(self) -> bool:
        return self.task == CF_TASK

    @property
    def n_total_spectra(self) -> int:
        return (
            self.run_info["train_samples"]
            + self.run_info["val_samples"]
            + self.run_info["test_samples"]
        )

    @property
    def splits(self) -> dict[str, np.ndarray]:
        return load_splits(
            self.cache_dir,
            self.n_total_spectra,
            self.run_info["test_samples"],
            strategy=self.cf_info.get("split_strategy"),
        )

    @property
    def spectra_path(self) -> Path:
        if self._spectra_path is None:
            self._spectra_path = resolve_spectra_cache(
                self.cache_dir, self.n_total_spectra, self.run_info,
            )
        return self._spectra_path

    @property
    def tokens_path(self) -> Path:
        if self._tokens_path is None:
            recorded = (self.run_info.get("line_tokens_path")
                        or self.cf_info.get("line_tokens_path"))
            if recorded:
                self._tokens_path = resolve_cache_path(
                    recorded, self.cache_dir, "line_tokens_*.h5",
                )
            else:
                # Nothing recorded (e.g. intensity run): match by spectrum count.
                chosen = None
                for cand in sorted(self.cache_dir.glob("line_tokens_*.h5")):
                    with h5py.File(cand, "r") as f:
                        if int(f.attrs.get("n_spectra", -1)) == self.n_total_spectra:
                            chosen = cand
                            break
                self._tokens_path = chosen or resolve_cache_path(
                    None, self.cache_dir, "line_tokens_*.h5",
                )
        return self._tokens_path

    def _read_token_meta(self) -> dict:
        with h5py.File(self.tokens_path, "r") as f:
            meta = {
                "n_lines": int(f.attrs["n_lines"]),
                "n_features": int(f.attrs["n_features"]),
                "feature_names": FEATURE_NAMES,
                "feature_mean": np.asarray(f.attrs["feature_mean"], dtype=np.float32),
                "feature_std": np.asarray(f.attrs["feature_std"], dtype=np.float32),
                "central_wavelength": f["central_wavelength"][:].astype(np.float32),
                # Provenance used by train_finetune.load_seed_module (token-layout
                # consistency of the CF seeds) and by the CF layer.
                "line_tokens_path": str(self.tokens_path),
                "line_dict_path": (str(f.attrs["line_dict_path"])
                                   if "line_dict_path" in f.attrs else None),
                "line_dict_hash": (str(f.attrs["line_dict_hash"])
                                   if "line_dict_hash" in f.attrs else None),
            }
        return meta

    @property
    def token_meta(self) -> dict | None:
        if self.embedding_type != "line_token_linear":
            return None
        if self._token_meta is None:
            self._token_meta = self._read_token_meta()
        return self._token_meta

    @property
    def concentrations(self) -> np.ndarray:
        if self._concentrations is None:
            with h5py.File(self.spectra_path, "r") as f:
                g = f["sample_table"]
                cols = [
                    np.asarray(g[name], dtype=np.float32) for name in self.element_names
                ]
                conc = np.stack(cols, axis=1)
            np.nan_to_num(conc, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
            self._concentrations = np.clip(conc, 0.0, 1.0)
        return self._concentrations

    @property
    def sample_table(self) -> pd.DataFrame:
        """Full ``sample_table`` of the spectra cache (element mass fractions,
        ids and, for physics_version-2 caches, the plasma columns)."""
        if self._sample_table is None:
            self._sample_table = read_sample_table(self.spectra_path)
        return self._sample_table

    @property
    def plasma_targets(self) -> dict[str, np.ndarray]:
        """Contract C2 aux targets for every spectrum of the cache."""
        if self._plasma_targets is None:
            self._plasma_targets = plasma_targets_from_table(self.sample_table)
        return self._plasma_targets

    @property
    def lod_vector(self) -> torch.Tensor:
        if self._lod is None:
            lod_map = self.run_info.get("element_lod") or {}
            default = float(self.run_info.get("default_lod", 1e-4))
            lod = np.array(
                [float(lod_map.get(name, default)) for name in self.element_names],
                dtype=np.float32,
            )
            self._lod = torch.from_numpy(lod)
        return self._lod

    def finetune_checkpoint(self) -> Path:
        best = self.run_dir / "checkpoints" / "best.ckpt"
        if best.is_file():
            return best
        ckpt = RunManager.from_existing_run(str(self.run_dir)).get_checkpoint_for_mode("finetune")
        if ckpt is None:
            raise FileNotFoundError(f"no checkpoint in {self.run_dir}")
        return Path(ckpt)

    # ── CF-specific assets ──
    @property
    def db_path(self) -> str:
        libs_cfg_path = self.run_info.get("libs_data_config")
        if libs_cfg_path and Path(libs_cfg_path).is_file():
            libs_cfg = yaml.safe_load(open(libs_cfg_path)) or {}
            return str((libs_cfg.get("paths") or {}).get("db", DEFAULT_DB_PATH))
        return DEFAULT_DB_PATH

    @property
    def lod_config_path(self) -> str | None:
        path = self.run_info.get("element_lod_config") or DEFAULT_LOD_CONFIG
        return path if Path(path).is_file() else None

    @property
    def cf_tables(self):
        """`cf.tables.CFTables` for this run's element order (contract C5)."""
        if self._cf_tables is None:
            from cf.tables import build_cf_tables
            self._cf_tables = build_cf_tables(
                self.element_names, self.db_path, lod_config_path=self.lod_config_path,
            )
        return self._cf_tables

    def _load_cf_seeds(self, cfg: dict) -> tuple[Any | None, Any | None, str]:
        """Frozen seed modules (binned C0 / detection presence) of a CF run.

        Returns (seed_binned, seed_detection, c0_source). Seeds are loaded
        through ``train_finetune.load_seed_module`` when it exists; otherwise
        they are skipped and ``c0_source`` falls back to ``uniform``.
        """
        c0_source = str(self.cf_info.get("c0_source") or "binned")
        try:
            from train_finetune import load_seed_module  # type: ignore
        except ImportError:
            load_seed_module = None
        if load_seed_module is None:
            print(f"[{self.label}] train_finetune.load_seed_module not available — "
                  "CF seeds skipped (c0_source=uniform)")
            return None, None, "uniform"

        # load_seed_module(run_dir, config, token_meta, *, strict_tokens,
        # expected_element_names): pass by keyword and only what it accepts.
        params = inspect.signature(load_seed_module).parameters
        extra = {k: v for k, v in {
            "strict_tokens": False,  # line-dictionary hash check is the real guard
            "expected_element_names": self.element_names,
        }.items() if k in params}
        seeds: dict[str, Any] = {}
        for key in ("seed_binned_run", "seed_detection_run"):
            run_dir = self.cf_info.get(key)
            if not run_dir:
                continue
            if not Path(run_dir).is_dir():
                print(f"[{self.label}] {key}={run_dir} not found — skipped")
                continue
            try:
                seed = load_seed_module(run_dir, config=cfg, token_meta=self.token_meta, **extra)
                seeds[key] = seed.to(self.device).eval()
                print(f"[{self.label}] loaded {key} from {run_dir}")
            except Exception as exc:
                print(f"[{self.label}] could not load {key} ({exc}) — skipped")
        seed_binned = seeds.get("seed_binned_run")
        seed_detection = seeds.get("seed_detection_run")
        if seed_binned is None and c0_source == "binned":
            c0_source = "uniform"
        return seed_binned, seed_detection, c0_source

    def _cf_module_kwargs(self, cfg: dict) -> dict[str, Any]:
        """Extra ``LIBSFinetuneModule`` kwargs for a ``cf_quantification`` run.

        Only keyword names accepted by the installed ``LIBSFinetuneModule``
        are passed, so the runner keeps working while the training-side API
        settles (candidates cover the plan's ``cf_tables / cf_cfg /
        seed_binned / seed_detection`` plus pure-physics, C0-source and
        line-dictionary variants).
        """
        seed_binned, seed_detection, c0_source = self._load_cf_seeds(cfg)
        pure = bool(self.cf_info.get("pure_physics", False))
        cf_cfg = dict(self.cf_info.get("cf_cfg") or cfg.get("finetune", {}).get("cf", {}) or {})
        # The module reads pure_physics / c0_source / line_dict_path from cf_cfg
        # (CF_CFG_DEFAULTS); keep them consistent with what could be loaded.
        cf_cfg["pure_physics"] = pure
        cf_cfg["c0_source"] = c0_source
        line_dict_path = (self.cf_info.get("line_dict_path") or cf_cfg.get("line_dict_path")
                          or (self.token_meta or {}).get("line_dict_path"))
        if line_dict_path and not Path(line_dict_path).is_file():
            local = self.cache_dir / Path(line_dict_path).name
            if local.is_file():
                line_dict_path = str(local)
            else:
                print(f"[{self.label}] line_dict_path {line_dict_path} not found — "
                      "isolation_score/forced buffers unavailable")
                line_dict_path = None
        cf_cfg["line_dict_path"] = line_dict_path
        candidates = {
            "cf_tables": self.cf_tables,
            "cf_cfg": cf_cfg,
            "seed_binned": seed_binned,
            "seed_detection": seed_detection,
            "cf_pure_physics": pure,
            "pure_physics": pure,
            "cf_c0_source": c0_source,
            "c0_source": c0_source,
            "cf_line_dict_path": line_dict_path,
            "line_dict_path": line_dict_path,
        }
        params = inspect.signature(LIBSFinetuneModule.__init__).parameters
        kwargs = {k: v for k, v in candidates.items() if k in params}
        dropped = sorted(set(candidates) - set(kwargs))
        if "cf_tables" not in kwargs:
            raise TypeError(
                "LIBSFinetuneModule does not accept `cf_tables`; the "
                f"{CF_TASK} task is not available in training/finetune.py",
            )
        print(f"[{self.label}] CF module kwargs: {sorted(kwargs)} "
              f"(pure_physics={pure}, c0_source={c0_source}; not accepted: {dropped})")
        return kwargs

    @property
    def module(self) -> LIBSFinetuneModule:
        if self._module is not None:
            return self._module

        cfg = yaml.safe_load(open(self.run_dir / "config.yaml"))
        token_meta = self.token_meta
        if self.embedding_type == "line_token_linear":
            if token_meta is None:
                raise ValueError("line_token_linear run requires token cache")
            cfg["data"]["n_bins"] = token_meta["n_lines"]
            cfg["model"]["max_seq_len"] = token_meta["n_lines"] + 1
        else:
            with h5py.File(self.spectra_path, "r") as f:
                n_bins = int(f["spectra"].shape[1])
            cfg["data"]["n_bins"] = n_bins
            cfg["model"]["max_seq_len"] = n_bins + 1

        encoder = build_encoder(cfg, self.run_info, token_meta)
        ckpt = self.finetune_checkpoint()
        state = _checkpoint_encoder_state(str(ckpt))
        enc_sd = encoder.state_dict()
        filtered = {
            k: v for k, v in state.items()
            if k in enc_sd and enc_sd[k].shape == v.shape
        }
        encoder.load_state_dict(filtered, strict=False)

        module_kwargs: dict[str, Any] = dict(
            encoder=encoder,
            task=self.run_info["task"],
            n_classes=cfg["data"]["n_classes"],
            n_elements=self.run_info["n_elements"],
            n_concentration_bins=self.run_info["n_concentration_bins"],
            pool=self.run_info["pool"],
            element_names=self.element_names,
            lod=self.lod_vector,
        )
        if self.is_cf:
            module_kwargs.update(self._cf_module_kwargs(cfg))
        module = LIBSFinetuneModule(**module_kwargs)
        ckpt_obj = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        sd = ckpt_obj["state_dict"] if "state_dict" in ckpt_obj else ckpt_obj
        mod_sd = module.state_dict()
        filtered_mod = {
            k: v for k, v in sd.items()
            if k in mod_sd and mod_sd[k].shape == v.shape
        }
        module.load_state_dict(filtered_mod, strict=False)
        self._module = module.to(self.device).eval()
        print(f"[{self.label}] Loaded module from {ckpt.name} ({self.embedding_type}, "
              f"task={self.task})")
        return self._module

    @torch.no_grad()
    def run_detection_inference(self, indices: np.ndarray) -> dict:
        """Detection outputs on the given global spectrum indices."""
        indices = np.asarray(indices, dtype=np.int64)
        module = self.module
        device = self.device
        lod = self.lod_vector.to(device)
        conc_all = self.concentrations[indices]

        preds, probs, reprs, targets, concs = [], [], [], [], []
        bs = self.batch_size
        if self.embedding_type == "intensity":
            bs = min(bs, 4)
        print(f"[{self.label}] Inference on {len(indices)} spectra ({device}, bs={bs})...")

        if self.embedding_type == "line_token_linear":
            with h5py.File(self.tokens_path, "r") as f:
                tok_ds, valid_ds = f["tokens"], f["fit_valid"]
                for start in range(0, len(indices), bs):
                    idx = indices[start:start + bs]
                    tokens = torch.from_numpy(tok_ds[idx].astype(np.float32))
                    valid = torch.from_numpy(valid_ds[idx].astype(np.uint8))
                    keep = valid.sum(dim=1) > 0
                    if not keep.any():
                        continue
                    batch = {
                        "tokens": tokens[keep].to(device),
                        "fit_valid": valid[keep].to(device),
                    }
                    out = module(batch)
                    conc_batch = torch.from_numpy(
                        conc_all[start:start + bs][keep.numpy()].astype(np.float32),
                    ).to(device)
                    preds.append(out["presence_pred"].cpu().numpy())
                    probs.append(out["presence_prob"].cpu().numpy())
                    targets.append(concentration_to_presence(conc_batch, lod).cpu().numpy())
                    reprs.append(out["representation"].float().cpu().numpy())
                    concs.append(conc_batch.cpu().numpy())
        else:
            with h5py.File(self.spectra_path, "r") as f:
                spec_ds = f["spectra"]
                for start in range(0, len(indices), bs):
                    idx = indices[start:start + bs]
                    spectra = torch.from_numpy(
                        spec_ds[idx].astype(np.float32),
                    ).to(device)
                    batch = {"spectrum": spectra}
                    out = module(batch)
                    conc_batch = torch.from_numpy(
                        conc_all[start:start + bs].astype(np.float32),
                    ).to(device)
                    preds.append(out["presence_pred"].cpu().numpy())
                    probs.append(out["presence_prob"].cpu().numpy())
                    targets.append(concentration_to_presence(conc_batch, lod).cpu().numpy())
                    reprs.append(out["representation"].float().cpu().numpy())
                    concs.append(conc_batch.cpu().numpy())

        return {
            "indices": indices,
            "preds": np.concatenate(preds, axis=0),
            "probs": np.concatenate(probs, axis=0),
            "targets": np.concatenate(targets, axis=0),
            "concentrations": np.concatenate(concs, axis=0),
            "representations": np.concatenate(reprs, axis=0),
        }

    # ── calibration-free quantification ──
    @staticmethod
    def _collect_cf_outputs(out: dict, store: dict[str, list]) -> None:
        """Append the CF forward outputs of one batch (contract C6) to `store`."""
        mapping = {
            "preds": "concentrations_pred",
            "number_fractions": "cf_number_fractions",
            "T": "cf_T",
            "log10_Ne": "cf_log10_Ne",
            "weights": "cf_weights",
            "tau0": "cf_tau0",
            "intercepts": "cf_intercepts",
            "censored": "cf_censored",
            "n_lines_used": "cf_n_lines_used",
            "used_mask": "cf_used_mask",
            "representations": "representation",
        }
        for key, src in mapping.items():
            arr = _to_numpy(out.get(src))
            if arr is not None:
                store.setdefault(key, []).append(arr)
        init = out.get("cf_init") or {}
        for k in ("T0", "log10_Ne0", "log10_Nl0"):
            arr = _to_numpy(init.get(k)) if isinstance(init, dict) else None
            if arr is not None:
                store.setdefault(f"init_{k}", []).append(arr)

    @torch.no_grad()
    def run_cf_inference(self, indices: np.ndarray) -> dict:
        """Calibration-free quantification on the given global spectrum indices.

        Both embedding branches feed the line tokens to the module: the
        ``line_token_linear`` branch as ``tokens``/``fit_valid`` only, the
        ``intensity`` branch as ``spectrum`` + ``tokens``/``fit_valid`` (the
        Saha–Boltzmann layer needs the raw tokens whatever the encoder sees).
        Spectra without a single valid Voigt fit are dropped.

        Returns a dict with ``preds`` (mass fractions [n, E]), ``targets``
        (true mass fractions [n, E]), ``concentrations`` (alias of targets),
        ``number_fractions`` [n, E], ``T`` [n] (K), ``log10_Ne`` [n],
        ``weights`` [n, L], ``tau0`` [n, L], ``intercepts`` [n, E],
        ``censored`` [n, E] bool, ``n_lines_used`` [n, E], ``fit_valid``
        [n, L] uint8, ``representations`` [n, d], ``indices`` [n] (kept global
        indices, in increasing order) and, when the module reports them,
        ``init_T0`` / ``init_log10_Ne0`` / ``init_log10_Nl0`` [n] and
        ``used_mask`` [n, L].
        """
        # h5py fancy indexing needs increasing indices; outputs follow this order.
        indices = np.unique(np.asarray(indices, dtype=np.int64))
        module = self.module
        device = self.device
        conc_all = self.concentrations[indices]

        store: dict[str, list] = {}
        kept_idx, targets, fit_valids = [], [], []
        bs = self.batch_size
        if self.embedding_type == "intensity":
            bs = min(bs, 4)
        print(f"[{self.label}] CF inference on {len(indices)} spectra ({device}, bs={bs})...")

        with h5py.File(self.tokens_path, "r") as f:
            tok_ds, valid_ds = f["tokens"], f["fit_valid"]
            spec_file = (
                h5py.File(self.spectra_path, "r")
                if self.embedding_type != "line_token_linear" else None
            )
            try:
                for start in range(0, len(indices), bs):
                    idx = indices[start:start + bs]
                    tokens = torch.from_numpy(tok_ds[idx].astype(np.float32))
                    valid = torch.from_numpy(valid_ds[idx].astype(np.uint8))
                    keep = valid.sum(dim=1) > 0
                    if not keep.any():
                        continue
                    keep_np = keep.numpy()
                    batch = {
                        "tokens": tokens[keep].to(device),
                        "fit_valid": valid[keep].to(device),
                    }
                    if spec_file is not None:
                        spectra = spec_file["spectra"][idx].astype(np.float32)[keep_np]
                        batch["spectrum"] = torch.from_numpy(spectra).to(device)
                    out = module(batch)
                    self._collect_cf_outputs(out, store)
                    targets.append(conc_all[start:start + bs][keep_np].astype(np.float32))
                    fit_valids.append(valid[keep].numpy())
                    kept_idx.append(idx[keep_np])
                    if (start // bs) % 10 == 9:
                        print(f"  {min(start + bs, len(indices))}/{len(indices)}")
            finally:
                if spec_file is not None:
                    spec_file.close()

        if not kept_idx:
            raise RuntimeError("no spectrum with a valid Voigt fit among the requested indices")
        result: dict[str, Any] = {
            k: np.concatenate(v, axis=0) for k, v in store.items()
        }
        result["targets"] = np.concatenate(targets, axis=0)
        result["concentrations"] = result["targets"]
        result["fit_valid"] = np.concatenate(fit_valids, axis=0)
        result["indices"] = np.concatenate(kept_idx, axis=0)
        # Contract keys are always present (None when the module omits them).
        for key in ("preds", "number_fractions", "T", "log10_Ne", "weights", "tau0",
                    "intercepts", "censored", "n_lines_used", "representations"):
            result.setdefault(key, None)
        if result["preds"] is None:
            raise KeyError("module output lacks `concentrations_pred` (contract C6)")
        if result["censored"] is not None:
            result["censored"] = result["censored"].astype(bool)
        print(f"  done: {result['preds'].shape[0]} spectra kept")
        return result


def load_pretrain_summary(pretrain_run: str | None) -> dict:
    if not pretrain_run:
        return {}
    path = Path(pretrain_run) / "run_info.yaml"
    if not path.is_file():
        return {"path": pretrain_run, "error": "run_info not found"}
    info = yaml.safe_load(open(path))
    return {
        "path": pretrain_run,
        "name": info.get("run_name", Path(pretrain_run).name),
        "experiment": info.get("experiment_name"),
        "embedding_type": info.get("embedding_type"),
        "pretrain_loss": info.get("pretrain_loss", "mse"),
        "epochs": info.get("epochs"),
        "n_lines": info.get("n_lines"),
        "best_val_loss": info.get("best_val_loss"),
    }
