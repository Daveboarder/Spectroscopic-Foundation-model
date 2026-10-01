"""
Timed end-to-end smoke run of the minerals pipeline:

    1. synthetic spectra   data.libs_pipeline.build_dataset_from_config (two-zone generator,
                           cache synthetic_cache_<key>.h5)
    2. pre-training        train_pretrain.py  (masked intensity prediction, self-supervised)
    3. classification      train_finetune.py --task classification (mineral identity when the
                           data config sets downstream.class_label: sample_type); the held-out
                           test split plays the "unknown minerals"

Every step runs as its own subprocess so the wall time, CPU time (utilisation =
CPU / wall tells how much parallel headroom a step has) and the sizes that drive
it (shots, pixels, epochs, batch) are recorded.  Per-epoch times are parsed from
the Lightning progress output.  Results:

    <out_dir>/timing.json       everything, machine-readable
    <out_dir>/timing.md         summary table
    <out_dir>/step<k>_*.log     full stdout/stderr of each step
    Outputs/pipeline_timings.csv   one row per step appended across runs
                                   (for comparing optimisations over time)

Usage:
    uv run python scripts/run_minerals_pipeline.py
    uv run python scripts/run_minerals_pipeline.py --config config/config_libs_smoke.yaml \\
        --libs_data_config config/libs_data_minerals_smoke.yaml --experiment minerals_smoke
    uv run python scripts/run_minerals_pipeline.py --skip_generate   # reuse the cache
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import resource
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EPOCH_RE = re.compile(r"Epoch (\d+): 100%\|[^|]*\| (\d+)/\d+ \[(\d+):(\d+)<")
RUN_DIR_RE = re.compile(r"Run directory: (\S+)")


def _hardware() -> dict:
    hw = dict(
        host=platform.node(),
        cpu_count=os.cpu_count(),
        python=platform.python_version(),
    )
    try:
        import torch

        hw["torch"] = torch.__version__
        hw["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            hw["gpu"] = torch.cuda.get_device_name(0)
            hw["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    except Exception as exc:  # pragma: no cover - torch missing
        hw["torch_error"] = str(exc)
    return hw


def _epoch_times(log_text: str) -> list[dict]:
    """Seconds per completed epoch from Lightning's progress bars (last line per epoch)."""
    found: dict[int, dict] = {}
    for m in EPOCH_RE.finditer(log_text):
        ep, n_batches, mm, ss = (int(m.group(i)) for i in range(1, 5))
        found[ep] = dict(epoch=ep, n_batches=n_batches, seconds=60 * mm + ss)
    return [found[k] for k in sorted(found)]


class Timer:
    """Wall + CPU time of a subprocess (children rusage delta)."""

    def __init__(self):
        self.r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
        self.t0 = time.perf_counter()

    def stop(self) -> dict:
        wall = time.perf_counter() - self.t0
        r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu = (r1.ru_utime - self.r0.ru_utime) + (r1.ru_stime - self.r0.ru_stime)
        return dict(
            wall_s=round(wall, 2),
            cpu_s=round(cpu, 2),
            cpu_utilisation=round(cpu / wall, 2) if wall > 0 else None,
            max_rss_children_gb=round(r1.ru_maxrss / 2**20, 2),
        )


def run_step(name: str, cmd: list[str], log_path: Path) -> tuple[dict, str]:
    print(f"\n=== {name} ===\n$ {' '.join(cmd)}")
    t = Timer()
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
    timing = t.stop()
    text = log_path.read_text(errors="replace")
    timing["returncode"] = proc.returncode
    print(
        f"    wall {timing['wall_s']:.1f} s, cpu {timing['cpu_s']:.1f} s "
        f"(utilisation {timing['cpu_utilisation']}), exit {proc.returncode}"
    )
    if proc.returncode != 0:
        tail = "\n".join(text.splitlines()[-25:])
        raise SystemExit(f"step '{name}' failed (exit {proc.returncode}); log tail:\n{tail}")
    return timing, text


GENERATE_SNIPPET = r"""
import json, sys, time, yaml, os
from pathlib import Path
sys.path.insert(0, __ROOT__)
from data.libs_pipeline import build_dataset_from_config
cfg = yaml.safe_load(open(__CFG__))
cache_dir = Path(cfg["paths"].get("cache_dir", "external_data/cache"))
before = set(p.name for p in cache_dir.glob("synthetic_cache_*.h5")) if cache_dir.is_dir() else set()
t0 = time.perf_counter()
ds = build_dataset_from_config(cfg)
dt = time.perf_counter() - t0
cache = Path(ds._cache_path())
n_two = int((ds.sample_table["plasma_model"] == "two_zone").sum()) if "plasma_model" in ds.sample_table else 0
print("GENERATE_RESULT " + json.dumps(dict(
    n_shots=int(len(ds)), n_px=int(ds.spectra.shape[1]), n_sample_types=int(ds.sample_table["sample_type_id"].nunique()),
    n_two_zone=n_two, cache_path=str(cache), cache_hit=cache.name in before, build_s=round(dt, 2),
    n_workers=int(cfg.get("generation", {}).get("n_workers", 1)), fine_step_nm=cfg.get("generation", {}).get("fine_step_nm"),
    plasma_model=cfg.get("generation", {}).get("plasma_model"))))
"""


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--config", default="config/config_libs_smoke.yaml", help="model / training YAML"
    )
    ap.add_argument("--libs_data_config", default="config/libs_data_minerals_smoke.yaml")
    ap.add_argument(
        "--experiment", default="minerals_smoke", help="experiment name suffix of the run folders"
    )
    ap.add_argument("--task", default="classification")
    ap.add_argument("--pool", default="cls_mean", choices=("cls", "mean", "cls_mean"))
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--out_dir", default=None, help="default Outputs/pipeline_<experiment>_<ts>")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--skip_generate",
        action="store_true",
        help="skip step 1 (pre-training builds/loads the cache anyway)",
    )
    ap.add_argument(
        "--pretrain_run_dir", default=None, help="skip step 2 and fine-tune from this run"
    )
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = (
        Path(args.out_dir)
        if args.out_dir
        else ROOT / "Outputs" / f"pipeline_{args.experiment}_{ts}"
    )
    out.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    libs_cfg = yaml.safe_load(open(ROOT / args.libs_data_config))
    model_cfg = yaml.safe_load(open(ROOT / args.config))
    report: dict = dict(
        timestamp=ts,
        experiment=args.experiment,
        config=args.config,
        libs_data_config=args.libs_data_config,
        task=args.task,
        hardware=_hardware(),
        steps={},
    )
    t_all = time.perf_counter()

    # ---- 1. synthetic spectra ------------------------------------------------
    if not args.skip_generate:
        snippet = GENERATE_SNIPPET.replace("__ROOT__", repr(str(ROOT))).replace(
            "__CFG__", repr(str(ROOT / args.libs_data_config))
        )
        timing, text = run_step(
            "1. synthetic spectra", [py, "-c", snippet], out / "step1_generate.log"
        )
        m = re.search(r"GENERATE_RESULT (\{.*\})", text)
        info = json.loads(m.group(1)) if m else {}
        if info:
            n = info["n_shots"]
            info["shots_per_s"] = round(n / info["build_s"], 2) if info["build_s"] > 0 else None
            print(
                f"    {n} shots x {info['n_px']} px, {info['n_sample_types']} minerals, "
                f"{'cache hit' if info['cache_hit'] else 'generated'} in {info['build_s']} s "
                f"({info['shots_per_s']} shots/s with {info['n_workers']} workers)"
            )
        report["steps"]["generate"] = dict(timing=timing, **info)

    # ---- 2. self-supervised pre-training --------------------------------------
    pretrain_dir = args.pretrain_run_dir
    if pretrain_dir is None:
        cmd = [
            py,
            "train_pretrain.py",
            "--config",
            args.config,
            "--libs_data_config",
            args.libs_data_config,
            "--experiment_name",
            args.experiment,
            "--runs_dir",
            args.runs_dir,
            "--seed",
            str(args.seed),
        ]
        timing, text = run_step("2. pre-training (MIP)", cmd, out / "step2_pretrain.log")
        m = RUN_DIR_RE.search(text)
        pretrain_dir = m.group(1) if m else None
        ri = yaml.safe_load(open(ROOT / pretrain_dir / "run_info.yaml")) if pretrain_dir else {}
        tr = re.search(r"Train: (\d+) samples \| Val: (\d+) samples \| n_bins: (\d+)", text)
        epochs = _epoch_times(text)
        pre = model_cfg.get("pretrain", {})
        info = dict(
            run_dir=pretrain_dir,
            n_train=int(tr.group(1)) if tr else None,
            n_val=int(tr.group(2)) if tr else None,
            n_bins=int(tr.group(3)) if tr else None,
            epochs=pre.get("epochs"),
            batch_size=pre.get("batch_size"),
            d_model=model_cfg.get("model", {}).get("d_model"),
            n_layers=model_cfg.get("model", {}).get("n_layers"),
            precision=model_cfg.get("device", {}).get("precision"),
            best_val_loss=ri.get("best_val_loss"),
            epoch_times=epochs,
            train_only_s=sum(e["seconds"] for e in epochs) if epochs else None,
        )
        if info["n_train"] and info["train_only_s"]:
            info["train_samples_per_s"] = round(
                info["n_train"] * len(epochs) / max(info["train_only_s"], 1e-9), 2
            )
        print(
            f"    run {pretrain_dir}; best val loss {info['best_val_loss']}; "
            f"epochs {[e['seconds'] for e in epochs]} s; overhead (data build, model init, checkpoints) "
            f"{timing['wall_s'] - (info['train_only_s'] or 0):.1f} s"
        )
        report["steps"]["pretrain"] = dict(timing=timing, **info)
    else:
        report["steps"]["pretrain"] = dict(run_dir=pretrain_dir, skipped=True)

    # ---- 3. classification of held-out minerals ----------------------------------
    cmd = [
        py,
        "train_finetune.py",
        "--config",
        args.config,
        "--libs_data_config",
        args.libs_data_config,
        "--pretrain_run_dir",
        pretrain_dir,
        "--task",
        args.task,
        "--pool",
        args.pool,
        "--experiment_name",
        args.experiment,
        "--runs_dir",
        args.runs_dir,
        "--seed",
        str(args.seed),
    ]
    timing, text = run_step(f"3. fine-tune ({args.task})", cmd, out / "step3_finetune.log")
    m = RUN_DIR_RE.search(text)
    finetune_dir = m.group(1) if m else None
    ri = yaml.safe_load(open(ROOT / finetune_dir / "run_info.yaml")) if finetune_dir else {}
    test = ri.get("test_results") or {}
    sp = re.search(r"Splits: train=(\d+), val=(\d+), test=(\d+)", text)
    cl = re.search(r"Classification labels: sample_type \((\d+) classes\)", text)
    epochs = _epoch_times(text)
    ft = model_cfg.get("finetune", {})
    info = dict(
        run_dir=finetune_dir,
        n_train=int(sp.group(1)) if sp else None,
        n_val=int(sp.group(2)) if sp else None,
        n_test=int(sp.group(3)) if sp else None,
        n_classes=int(cl.group(1)) if cl else libs_cfg.get("downstream", {}).get("n_clusters"),
        epochs=ft.get("epochs"),
        batch_size=ft.get("batch_size"),
        epoch_times=epochs,
        train_only_s=sum(e["seconds"] for e in epochs) if epochs else None,
        test_metrics={
            k: (float(v) if isinstance(v, (int, float)) else v)
            for k, v in test.items()
            if isinstance(v, (int, float))
        },
    )
    acc = info["test_metrics"].get("test/accuracy")
    print(
        f"    run {finetune_dir}; splits {info['n_train']}/{info['n_val']}/{info['n_test']}, "
        f"{info['n_classes']} classes; test metrics {info['test_metrics']}"
    )
    report["steps"]["finetune"] = dict(timing=timing, **info)

    # ---- report ---------------------------------------------------------------
    report["total_wall_s"] = round(time.perf_counter() - t_all, 2)
    (out / "timing.json").write_text(json.dumps(report, indent=2, default=str))

    rows = []
    for k, st in report["steps"].items():
        tm = st.get("timing", {})
        size = {
            "generate": f"{st.get('n_shots')} shots x {st.get('n_px')} px, {st.get('n_workers')} workers"
            + (" (cache hit)" if st.get("cache_hit") else ""),
            "pretrain": f"{st.get('n_train')} train / {st.get('n_val')} val, {st.get('epochs')} epochs, batch {st.get('batch_size')}",
            "finetune": f"{st.get('n_train')}/{st.get('n_val')}/{st.get('n_test')} split, {st.get('n_classes')} classes, {st.get('epochs')} epochs",
        }.get(k, "")
        rows.append(
            (
                k,
                tm.get("wall_s"),
                tm.get("cpu_s"),
                tm.get("cpu_utilisation"),
                st.get("train_only_s"),
                size,
            )
        )
    md = [
        f"# Pipeline timing — {args.experiment} ({ts})",
        "",
        f"hardware: {report['hardware']}",
        "",
        "| step | wall s | cpu s | cpu util | train-only s | size |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        md.append(
            f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4] if r[4] is not None else ''} | {r[5]} |"
        )
    md.append(f"| **total** | {report['total_wall_s']} | | | | |")
    if acc is not None:
        md += [
            "",
            f"held-out classification accuracy: **{acc:.3f}** ({info['n_test']} test spectra, {info['n_classes']} minerals)",
        ]
    (out / "timing.md").write_text("\n".join(md) + "\n")

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
        for r in rows:
            w.writerow([ts, args.experiment, *r, report["hardware"].get("gpu")])

    print("\n" + "\n".join(md))
    print(f"\n[results] {out}  (total {report['total_wall_s']:.1f} s; csv {csv_path})")


if __name__ == "__main__":
    main()
