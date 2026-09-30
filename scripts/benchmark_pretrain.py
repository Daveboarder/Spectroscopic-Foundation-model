"""
Throughput benchmark of self-supervised pre-training (masked intensity
prediction) on real LIGHTIGO spectra with a full-size model config.

Loads ``--n_spectra`` spectra from one LIGHTIGO HDF5 map (evenly spaced over the
map, invalid shots skipped), unit-normalises them, and trains the same model /
data module / Lightning trainer that ``train_pretrain.py`` builds from
``--config`` for ``--epochs`` epochs.  A callback times every optimiser step
(CUDA-synchronised) so the result separates:

    load_s             reading the spectra from the file share
    setup_s            model + trainer construction
    train step times   per batch; the first ``--warmup_steps`` are excluded
                       (CUDA / cuDNN warm-up, kernel selection)
    spectra_per_s      steady-state training throughput = batch / median step time
    epoch_spectra_per_s  whole training epochs incl. data loading and logging
    val_s              validation pass time
    peak_gpu_mem_gb    torch.cuda.max_memory_allocated

and extrapolates the time for one epoch over ``--extrapolate`` spectra
(default: the 220,349,058 spectra of the Running_projects inventory).

Results: Outputs/benchmark_pretrain_<ts>/benchmark.json and a row appended to
Outputs/pipeline_timings.csv (step = "pretrain_benchmark").

Usage:
    uv run python scripts/benchmark_pretrain.py \\
        --h5 "/mnt/data/projects/Running_projects/24_0044_Space_plants/Strawberry plants/sample2B_1ablace_2025_02_18_Jahodnik_275x184_7mJ_100GD.h5"
    uv run python scripts/benchmark_pretrain.py --h5 ... --config config/config_libs_a100.yaml --batch_size 16
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pytorch_lightning as pl
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.libs_pipeline import unit_norm  # noqa: E402
from train_pretrain import (  # noqa: E402
    build_pretrain_discretizers,
    create_model,
    load_config,
    pretrain_loss_type,
)
from training.pretrain import LIBSPretrainModule, PretrainDataModule  # noqa: E402

INVENTORY_SPECTRA = 220_349_058  # Outputs/h5_files_spectra_2026-09-25_summary.txt


class StepTimer(pl.Callback):
    """Wall time of every training batch (CUDA-synchronised) and of validation."""

    def __init__(self):
        self.step_s: list[float] = []
        self.val_s: list[float] = []
        self.epoch_s: list[float] = []
        self._t = self._tv = self._te = 0.0

    @staticmethod
    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def on_train_epoch_start(self, trainer, pl_module):
        self._sync()
        self._te = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module):
        self._sync()
        self.epoch_s.append(time.perf_counter() - self._te)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._sync()
        self._t = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._sync()
        self.step_s.append(time.perf_counter() - self._t)

    def on_validation_epoch_start(self, trainer, pl_module):
        self._sync()
        self._tv = time.perf_counter()

    def on_validation_epoch_end(self, trainer, pl_module):
        self._sync()
        self.val_s.append(time.perf_counter() - self._tv)


def load_spectra(path: str, n: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """``n`` valid spectra evenly spaced over the first measurement of a LIGHTIGO file."""
    with h5py.File(path, "r") as f:
        key = sorted(f["measurements"])[0]
        g = f["measurements"][key]["libs"]
        wl = g["calibration"][...].astype(np.float64)
        n_total = g["data"].shape[0]
        valid = np.arange(n_total)
        if "metadata" in g and "Invalid" in g["metadata"]:
            valid = np.nonzero(g["metadata"]["Invalid"][...].ravel() == 0)[0]
        idx = np.unique(np.linspace(0, valid.size - 1, min(n, valid.size)).round().astype(int))
        idx = np.sort(valid[idx])
        X = g["data"][idx, :].astype(np.float32)
    info = dict(
        measurement=key, n_in_file=int(n_total), n_valid=int(valid.size), n_loaded=int(len(idx))
    )
    return X, wl, info


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--h5", required=True, help="LIGHTIGO HDF5 map")
    ap.add_argument(
        "--config", default="config/config_libs_4090.yaml", help="full-size model config"
    )
    ap.add_argument("--n_spectra", type=int, default=2000)
    ap.add_argument("--val_fraction", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=2, help="epoch 1 includes warm-up")
    ap.add_argument("--warmup_steps", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=None, help="override pretrain.batch_size")
    ap.add_argument(
        "--precision", default=None, help="override device.precision, e.g. 32 or bf16-mixed"
    )
    ap.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    ap.add_argument(
        "--extrapolate",
        type=int,
        default=INVENTORY_SPECTRA,
        help="spectra for the epoch-time estimate",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = ROOT / "Outputs" / f"benchmark_pretrain_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")

    # ---- data ---------------------------------------------------------------
    t0 = time.perf_counter()
    X, wl, finfo = load_spectra(args.h5, args.n_spectra)
    load_s = time.perf_counter() - t0
    X = np.stack([unit_norm(x) for x in X]).astype(np.float32)
    n_val = max(1, int(round(args.val_fraction * len(X))))
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(X))
    val, train = X[perm[:n_val]], X[perm[n_val:]]
    print(
        f"Loaded {len(X)} spectra x {X.shape[1]} px from {Path(args.h5).name} in {load_s:.1f} s "
        f"({X.nbytes / 2**20:.0f} MB, {X.nbytes / 2**20 / max(load_s, 1e-9):.0f} MB/s); "
        f"train {len(train)}, val {len(val)}"
    )

    # ---- model (same construction as train_pretrain.py) --------------------------
    t1 = time.perf_counter()
    config = load_config(str(ROOT / args.config))
    config["data"]["n_bins"] = X.shape[1]
    config["model"]["max_seq_len"] = X.shape[1] + 1
    pre = config["pretrain"]
    if args.batch_size:
        pre["batch_size"] = args.batch_size
    if args.precision:
        config["device"]["precision"] = args.precision
    pre["epochs"] = args.epochs
    model = create_model(config)
    loss_type = pretrain_loss_type(config)
    idisc, fdisc = build_pretrain_discretizers(config)
    module = LIBSPretrainModule(
        model=model,
        learning_rate=pre["learning_rate"],
        weight_decay=pre["weight_decay"],
        warmup_epochs=min(pre["warmup_epochs"], args.epochs),
        max_epochs=args.epochs,
        min_lr=pre["min_lr"],
        loss_type=loss_type,
        intensity_discretizer=idisc,
        fwhm_discretizer=fdisc,
    )
    block_sizes = pre.get("block_sizes") or [pre.get("contiguous_mask_size", 50)]
    dm = PretrainDataModule(
        train_spectra=train,
        val_spectra=val,
        batch_size=pre["batch_size"],
        mask_ratio=pre["mask_ratio"],
        contiguous_masking=pre.get("contiguous_masking", False),
        block_sizes=block_sizes,
        peak_bias_enabled=pre.get("peak_bias_enabled", False),
        peak_bias_ratio=pre.get("peak_bias_ratio", 0.5),
        peak_threshold=pre.get("peak_threshold", 0.2),
        num_workers=args.num_workers,
    )
    timer = StepTimer()
    accumulate = pre.get("accumulate_grad_batches") or 1
    trainer = pl.Trainer(
        accelerator=config["device"]["accelerator"],
        devices=1,
        precision=config["device"]["precision"],
        max_epochs=args.epochs,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[timer],
        gradient_clip_val=1.0,
        accumulate_grad_batches=accumulate,
        num_sanity_val_steps=0,
    )
    setup_s = time.perf_counter() - t1
    m = config["model"]
    print(
        f"Model: d_model {m['d_model']}, heads {m['n_heads']}, layers {m['n_layers']}, d_ff {m['d_ff']}, "
        f"{model.num_parameters:,} parameters; batch {pre['batch_size']} x accumulate {accumulate}, "
        f"precision {config['device']['precision']}, loss {loss_type}"
    )

    # ---- train ------------------------------------------------------------------
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t2 = time.perf_counter()
    trainer.fit(module, dm)
    fit_s = time.perf_counter() - t2

    bs = pre["batch_size"]
    steps = timer.step_s[args.warmup_steps :] or timer.step_s
    med = statistics.median(steps)
    sps = bs / med
    epoch_sps = [len(train) / e for e in timer.epoch_s]
    est_s = args.extrapolate / sps
    res = dict(
        timestamp=ts,
        h5=args.h5,
        file=finfo,
        config=args.config,
        model=dict(
            d_model=m["d_model"],
            n_heads=m["n_heads"],
            n_layers=m["n_layers"],
            d_ff=m["d_ff"],
            parameters=int(model.num_parameters),
            seq_len=int(X.shape[1] + 1),
        ),
        batch_size=bs,
        accumulate_grad_batches=accumulate,
        precision=str(config["device"]["precision"]),
        num_workers=args.num_workers,
        n_train=int(len(train)),
        n_val=int(len(val)),
        epochs=args.epochs,
        load_s=round(load_s, 2),
        setup_s=round(setup_s, 2),
        fit_s=round(fit_s, 2),
        n_steps=len(timer.step_s),
        step_s_median=round(med, 4),
        step_s_p10=round(float(np.percentile(steps, 10)), 4),
        step_s_p90=round(float(np.percentile(steps, 90)), 4),
        first_step_s=round(timer.step_s[0], 3) if timer.step_s else None,
        spectra_per_s=round(sps, 2),
        epoch_s=[round(e, 2) for e in timer.epoch_s],
        epoch_spectra_per_s=[round(e, 2) for e in epoch_sps],
        val_s=[round(v, 2) for v in timer.val_s],
        val_spectra_per_s=[round(len(val) / v, 2) for v in timer.val_s if v > 0],
        peak_gpu_mem_gb=(
            round(torch.cuda.max_memory_allocated() / 2**30, 2)
            if torch.cuda.is_available()
            else None
        ),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        host=platform.node(),
        cpu_count=os.cpu_count(),
        torch=torch.__version__,
        extrapolate_spectra=args.extrapolate,
        extrapolated_epoch_hours=round(est_s / 3600, 1),
        extrapolated_epoch_days=round(est_s / 86400, 1),
    )
    (out / "benchmark.json").write_text(json.dumps(res, indent=2))

    csv_path = ROOT / "Outputs" / "pipeline_timings.csv"
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(
                [
                    "timestamp",
                    "experiment",
                    "step",
                    "wall_s",
                    "cpu_s",
                    "cpu_utilisation",
                    "train_only_s",
                    "size",
                    "gpu",
                ]
            )
        size = (
            f"{len(train)} train / {len(val)} val real LIGHTIGO, {args.epochs} epochs, batch {bs}x{accumulate}, "
            f"{config['device']['precision']}, d{m['d_model']} L{m['n_layers']}; {sps:.1f} spectra/s steady"
        )
        w.writerow(
            [
                ts,
                "benchmark_pretrain",
                "pretrain_benchmark",
                round(fit_s, 2),
                "",
                "",
                round(sum(timer.epoch_s), 2),
                size,
                res["gpu"],
            ]
        )

    print(
        "\n".join(
            [
                "",
                f"steady-state training: {sps:.1f} spectra/s (median step {med * 1000:.0f} ms for batch {bs}; "
                f"p10-p90 {res['step_s_p10'] * 1000:.0f}-{res['step_s_p90'] * 1000:.0f} ms; first step {res['first_step_s']} s)",
                f"whole epochs:          {', '.join(f'{e:.1f}' for e in epoch_sps)} spectra/s "
                f"({', '.join(f'{e:.1f}' for e in timer.epoch_s)} s per epoch of {len(train)})",
                f"validation:            {', '.join(f'{v:.1f}' for v in res['val_spectra_per_s'])} spectra/s",
                f"file read:             {load_s:.1f} s for {len(X)} spectra; setup {setup_s:.1f} s; peak GPU memory {res['peak_gpu_mem_gb']} GB",
                f"extrapolated epoch over {args.extrapolate:,} spectra: {res['extrapolated_epoch_hours']:,} h "
                f"= {res['extrapolated_epoch_days']:,} days",
                f"[results] {out / 'benchmark.json'}  (+ row in {csv_path})",
            ]
        )
    )


if __name__ == "__main__":
    main()
