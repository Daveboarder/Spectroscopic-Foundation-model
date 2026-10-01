"""
Mineral map of a measured LIBS raster with a fine-tuned classification run
(``--task classification``, ``class_label: sample_type``) of either encoder type:

* ``spectral_patch``: every selected shot is unit-normalised like the synthetic
  spectra (``libs_pipeline.unit_norm``) and classified directly (GPU only, no
  fitting); the encoder is rebuilt on the map's own wavelength axis.
* ``line_token_linear``: every shot is additionally Voigt-fitted on the run's
  line dictionary with the line-embedding config's fit settings
  (``data/line_features``, CPU worker pool) and turned into 14-channel tokens
  (static channels copied from the run's synthetic token cache).

No spectral-response correction, no saturation handling: the model is applied
"as is".  The raw peak (counts) of every shot is stored, so ``--gate_label
epoxid --gate_peak_counts 3000`` can assign the weak shots to one class and
exclude that class from the rest when the map is drawn (also with --plot_only).

HDF5 layout (LIGHTIGO / Avantes raster): ``measurements/<key>/libs/{calibration,
data, metadata/{X, Y, X_pos, Y_pos, Invalid}}``; ``X``/``Y`` are grid indices,
``X_pos``/``Y_pos`` stage positions in mm.  ``--stride s`` keeps the shots with
``X % s == 0 and Y % s == 0``.

Resumable: each slab of ``--slab`` rows is written to
``<out_dir>/blocks/block_<start>.npz`` and skipped when present.
Outputs: ``predictions.csv`` (one row per shot), ``mineral_map.png``,
``confidence_map.png``, ``summary.json``.

Usage:
    uv run python scripts/predict_mineral_map.py \\
        --h5 /mnt/data/projects/Running_projects/24_0010_Minerals_classification/Data/MAR1A/LIBS/Mar1A.h5 \\
        --run_dir runs/finetune_<ts>_minerals_patch_cls --stride 1
    # redraw with a gate (after all blocks exist):
    #   add --plot_only --gate_label epoxid --gate_peak_counts 3000 --out_dir <same dir>
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

# one BLAS thread per fit worker; the pool already uses every core
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.canonical import patch_input, unit_norm_rows  # noqa: E402
from data.line_features import FEAT_VALID, fit_line_in_spectrum  # noqa: E402
from data.line_features import N_FEATURES as N_FIT  # noqa: E402
from data.line_tokenization import F_DELTA, F_FWHM, F_MAX_I, F_R2, F_RMSE  # noqa: E402

# fit-feature column -> token channel (line_tokenization.build_line_tokens_cache)
FIT_TO_TOKEN = {0: F_MAX_I, 1: F_FWHM, 2: F_R2, 3: F_DELTA, 4: F_RMSE}
FIT_KEYS = (
    "window_nm",
    "gamma_init",
    "sigma_init",
    "r2_min",
    "baseline",
    "max_delta_nm",
    "max_fwhm_nm",
)
MODES = {"line_token_linear": "tokens", "spectral_patch": "patch"}

_w: dict = {}


def _init_worker(wavelength: np.ndarray, centres: np.ndarray, fit_cfg: dict) -> None:
    _w.update(wl=wavelength, centres=centres, cfg=fit_cfg)


def _fit_one(spectrum: np.ndarray) -> np.ndarray:
    """[n_lines, 6] fit features of one unit-normalised spectrum."""
    c = _w["cfg"]
    out = np.zeros((_w["centres"].size, N_FIT), dtype=np.float32)
    for j, centre in enumerate(_w["centres"]):
        out[j] = fit_line_in_spectrum(
            spectrum,
            _w["wl"],
            float(centre),
            c["window_nm"],
            c["gamma_init"],
            c["sigma_init"],
            c["r2_min"],
            baseline=c.get("baseline", "none"),
            max_delta_nm=c.get("max_delta_nm"),
            max_fwhm_nm=c.get("max_fwhm_nm"),
        )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
def _classifier_module(
    run_dir: Path, runner, n_classes: int, wavelength=None, checkpoint: str = "best"
):
    """LIBSFinetuneModule of the run with its ``checkpoint`` ('best': best.ckpt, selected on
    synthetic validation accuracy; 'last': last.ckpt) and a strict key check."""
    from analyze_attention_importance import _checkpoint_encoder_state, build_encoder
    from training.finetune import LIBSFinetuneModule

    cfg = yaml.safe_load(open(run_dir / "config.yaml"))
    cfg["data"]["n_classes"] = n_classes
    if runner.embedding_type == "line_token_linear":
        meta = runner.token_meta
        cfg["data"]["n_bins"] = meta["n_lines"]
        cfg["model"]["max_seq_len"] = meta["n_lines"] + 1
        encoder = build_encoder(cfg, runner.run_info, meta)
    else:
        encoder = build_encoder(cfg, runner.run_info, None, wavelength=wavelength)
    ckpt = runner.finetune_checkpoint(prefer=checkpoint)
    enc_state = _checkpoint_encoder_state(str(ckpt))
    missing, _ = encoder.load_state_dict(enc_state, strict=False)
    missing = [k for k in missing if not k.startswith("mip_")]  # pretrain-only MIP heads
    if missing:
        raise RuntimeError(f"encoder keys missing from {ckpt}: {missing[:5]}")
    module = LIBSFinetuneModule(
        encoder=encoder,
        task="classification",
        n_classes=n_classes,
        n_elements=runner.run_info["n_elements"],
        n_concentration_bins=runner.run_info["n_concentration_bins"],
        pool=runner.run_info["pool"],
        element_names=runner.element_names,
        lod=runner.lod_vector,
    )
    sd = torch.load(str(ckpt), map_location="cpu", weights_only=False)["state_dict"]
    missing, _ = module.load_state_dict(sd, strict=False)
    missing = [
        k
        for k in missing
        if "classification_head" in k or (k.startswith("encoder.") and ".mip_" not in k)
    ]
    if missing:
        raise RuntimeError(f"classifier keys missing from {ckpt}: {missing[:5]}")
    return module.to(runner.device).eval()


def load_classifier(
    run_dir: Path, device: str, map_wavelength: np.ndarray, checkpoint: str = "best"
) -> dict:
    """{mode, module, class_names, table, dataset, runner[, static | input_spec]} of a
    classification run; spectral_patch encoders are rebuilt on ``map_wavelength`` and
    their input spec (preprocess + canonical constants) comes from run_info."""
    from data.libs_pipeline import build_dataset_from_config
    from models.spectral_patch_embedding import axis_signature
    from publication.inference_runner import FinetuneInferenceRunner

    runner = FinetuneInferenceRunner(run_dir, device=device)
    mode = MODES.get(runner.embedding_type)
    if runner.task != "classification" or mode is None:
        raise ValueError(
            f"need a line_token_linear or spectral_patch classification run, got "
            f"{runner.task}/{runner.embedding_type}"
        )

    # Class index = position in np.unique(sample_type_id) (train_finetune.py,
    # class_label: sample_type). config.yaml keeps the placeholder n_classes, so
    # the class count must come from the data, not from the config. The dataset is
    # rebuilt from the run's data config so that measured extra classes
    # (extra_spectra, e.g. epoxid), which are not in the synthetic cache file, count.
    libs_cfg = yaml.safe_load(open(ROOT / runner.run_info["libs_data_config"]))
    libs_cfg.setdefault("generation", {})["verbose"] = False
    ds = build_dataset_from_config(libs_cfg)
    table = ds.sample_table
    ids = np.unique(table["sample_type_id"].astype(str).to_numpy())
    id_to_name = dict(
        zip(table["sample_type_id"].astype(str), table["sample_type_name"].astype(str))
    )
    class_names = [id_to_name[i] for i in ids]

    out = {"mode": mode, "class_names": class_names, "table": table, "dataset": ds}
    out["runner"] = runner
    out["checkpoint"] = checkpoint
    if mode == "tokens":
        out["module"] = _classifier_module(run_dir, runner, len(class_names), checkpoint=checkpoint)
        with h5py.File(runner.tokens_path, "r") as f:
            out["static"] = f["tokens"][0].astype(np.float32)  # identical for every row
        return out

    info = runner.run_info.get("spectral_patch") or {}
    train_axis = info.get("axis") or {}
    map_axis = axis_signature(map_wavelength)
    if train_axis.get("n_segments") not in (None, map_axis["n_segments"]):
        raise ValueError(
            f"map axis has {map_axis['n_segments']} detector segments, the model was "
            f"trained on {train_axis['n_segments']}"
        )
    if train_axis.get("md5") != map_axis["md5"]:
        print(
            f"note: map axis {map_axis} differs from the training axis {train_axis}; "
            "the windows are resampled in nm"
        )
    out["input_spec"] = {
        "preprocess": info.get("preprocess", "none"),
        "canonical": info.get("canonical"),
    }
    out["module"] = _classifier_module(
        run_dir, runner, len(class_names), wavelength=map_wavelength, checkpoint=checkpoint
    )
    same_axis = train_axis.get("md5") == map_axis["md5"]
    out["module_train_axis"] = (
        out["module"]
        if same_axis
        else _classifier_module(
            run_dir, runner, len(class_names), wavelength=np.asarray(ds.wavelength),
            checkpoint=checkpoint,
        )
    )  # fmt: skip
    return out


@torch.no_grad()
def classify_tokens(module, tokens: np.ndarray, valid: np.ndarray, device: str, bs: int = 256):
    """Softmax probabilities [n, n_classes]; rows without any valid line -> NaN."""
    out = []
    for s in range(0, len(tokens), bs):
        t = torch.from_numpy(tokens[s : s + bs]).to(device)
        v = torch.from_numpy(valid[s : s + bs]).to(device)
        logits = module({"tokens": t, "fit_valid": v})["class_logits"].float()
        p = torch.softmax(logits, dim=-1).cpu().numpy()
        p[valid[s : s + bs].sum(1) == 0] = np.nan
        out.append(p)
    return np.concatenate(out)


@torch.no_grad()
def classify_spectra(module, spectra: np.ndarray, device: str, bs: int = 512):
    """Softmax probabilities [n, n_classes] of spectral_patch model input (patch_input)."""
    out = []
    for s in range(0, len(spectra), bs):
        x = torch.from_numpy(np.ascontiguousarray(spectra[s : s + bs])).to(device)
        logits = module({"spectrum": x})["class_logits"].float()
        out.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 0), np.float32)


def sanity_check(clf: dict, device: str) -> float:
    """Accuracy on the run's test split (must match run_info test/accuracy)."""
    runner, table = clf["runner"], clf["table"]
    test = np.sort(runner.splits["test"])
    if clf["mode"] == "tokens":
        with h5py.File(runner.tokens_path, "r") as f:
            tok = f["tokens"][test].astype(np.float32)
            val = f["fit_valid"][test].astype(np.uint8)
        probs = classify_tokens(clf["module"], tok, val, device)
    else:
        ds = clf["dataset"]
        spectra = patch_input(
            ds.spectra[test], ds.wavelength, getattr(ds, "units", "unit_norm"), clf["input_spec"]
        )
        probs = classify_spectra(clf["module_train_axis"], spectra, device)
    all_ids = table["sample_type_id"].astype(str).to_numpy()
    truth = np.searchsorted(np.unique(all_ids), all_ids[test])
    return float((probs.argmax(1) == truth).mean())


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────
def mineral_colours(names_by_freq: list[str]) -> dict[str, tuple]:
    import matplotlib.pyplot as plt

    pal = [
        *plt.get_cmap("tab20").colors,
        *plt.get_cmap("tab20b").colors,
        *plt.get_cmap("tab20c").colors,
        *plt.get_cmap("Set1").colors,
    ]
    return {n: pal[i % len(pal)] for i, n in enumerate(names_by_freq)}


def plot_maps(df: pd.DataFrame, out_dir: Path, title: str, min_frac: float) -> dict:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    ok = df["predicted"].notna()
    counts = df.loc[ok, "predicted"].value_counts()
    frac = counts / counts.sum()
    shown = list(frac.index[frac >= min_frac])
    other = [n for n in frac.index if n not in shown]
    label = df["predicted"].where(~df["predicted"].isin(other), "other").where(ok, "no prediction")
    order = shown + (["other"] if other else []) + (["no prediction"] if (~ok).any() else [])
    colours = mineral_colours(shown)
    colours["other"] = (0.45, 0.45, 0.45)
    colours["no prediction"] = (1.0, 1.0, 1.0)

    xs, ys = np.unique(df["X"]), np.unique(df["Y"])
    xi, yi = np.searchsorted(xs, df["X"]), np.searchsorted(ys, df["Y"])
    grid = np.full((ys.size, xs.size), -1, dtype=int)
    grid[yi, xi] = pd.Categorical(label, categories=order).codes
    conf = np.full((ys.size, xs.size), np.nan)
    conf[yi, xi] = df["confidence"]

    # stage positions (mm) of the grid columns / rows, for the axes
    xp = df.groupby("X")["X_pos"].median().reindex(xs).to_numpy()
    yp = df.groupby("Y")["Y_pos"].median().reindex(ys).to_numpy()
    extent = [xp.min(), xp.max(), yp.min(), yp.max()]
    flip = yp[0] > yp[-1]  # Y index running against Y_pos
    img, cimg = (grid[::-1], conf[::-1]) if flip else (grid, conf)

    cmap = ListedColormap([(0.8, 0.8, 0.8)] + [colours[n] for n in order])
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(
        img + 1,
        cmap=cmap,
        vmin=0,
        vmax=len(order),
        origin="lower",
        extent=extent,
        interpolation="nearest",
    )
    ax.set_xlabel("X position [mm]")
    ax.set_ylabel("Y position [mm]")
    ax.set_title(title)
    ax.set_aspect("equal")
    n_all = len(df)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=colours[n], ec="0.5", lw=0.5) for n in order]
    labels = []
    for n in order:
        k = int((label == n).sum())
        labels.append(f"{n} ({100 * k / n_all:.1f} %)")
    ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        fontsize=8,
        frameon=False,
        title=f"predicted mineral (n = {n_all:,})",
    )
    fig.savefig(out_dir / "mineral_map.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(
        cimg, cmap="viridis", vmin=0, vmax=1, origin="lower", extent=extent, interpolation="nearest"
    )
    ax.set_xlabel("X position [mm]")
    ax.set_ylabel("Y position [mm]")
    ax.set_title(f"{title}\nsoftmax confidence of the predicted mineral", fontsize=10)
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, shrink=0.8, label="max class probability")
    fig.tight_layout()
    fig.savefig(out_dir / "confidence_map.png", dpi=200)
    plt.close(fig)
    return {n: round(float(frac.get(n, 0.0)), 5) for n in frac.index}


def apply_gate(
    probs: np.ndarray, peak: np.ndarray, class_names: list[str], label: str, counts: float
) -> tuple[np.ndarray, np.ndarray]:
    """Shots with raw peak < ``counts`` -> class ``label`` (gated); the others
    have that class removed and their probabilities renormalised.
    Returns (probs, gated mask)."""
    if label not in class_names:
        raise ValueError(f"--gate_label {label!r} is not a class of the run")
    j = class_names.index(label)
    gated = peak < counts
    p = probs.copy()
    p[gated] = 0.0
    p[gated, j] = 1.0
    keep = ~gated
    p[keep, j] = 0.0
    s = p[keep].sum(axis=1, keepdims=True)
    p[keep] = np.divide(p[keep], s, out=np.zeros_like(p[keep]), where=s > 0)
    return p, gated


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--h5", required=True)
    ap.add_argument("--run_dir", required=True, help="finetune run (task classification)")
    ap.add_argument("--measurement", default=None, help="measurement key (default: first)")
    ap.add_argument("--stride", type=int, default=2, help="keep X %% s == 0 and Y %% s == 0")
    ap.add_argument("--slab", type=int, default=8192, help="HDF5 rows read per block")
    ap.add_argument("--workers", type=int, default=22, help="Voigt-fit workers (token runs)")
    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "--out_dir", default=None, help="default Outputs/mineral_map_<h5 stem>_s<stride>"
    )
    ap.add_argument(
        "--min_frac",
        type=float,
        default=0.002,
        help="minerals below this pixel fraction are drawn as 'other'",
    )
    ap.add_argument(
        "--gate_label", default=None, help="class given to shots below --gate_peak_counts"
    )
    ap.add_argument(
        "--gate_peak_counts", type=float, default=None, help="raw peak threshold of the gate"
    )
    ap.add_argument("--plot_only", action="store_true")
    ap.add_argument(
        "--max_blocks", type=int, default=None, help="process at most N blocks (testing)"
    )
    ap.add_argument(
        "--checkpoint",
        choices=("best", "last"),
        default="best",
        help="best.ckpt (selected on synthetic validation accuracy) or last.ckpt",
    )
    args = ap.parse_args()
    if (args.gate_label is None) != (args.gate_peak_counts is None):
        ap.error("--gate_label and --gate_peak_counts go together")

    h5_path = Path(args.h5)
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir or ROOT / "Outputs" / f"mineral_map_{h5_path.stem}_s{args.stride}")
    blocks = out_dir / "blocks"
    blocks.mkdir(parents=True, exist_ok=True)
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )

    run_info = yaml.safe_load(open(run_dir / "run_info.yaml"))
    mode = MODES.get(run_info.get("embedding_type"))

    with h5py.File(h5_path, "r") as f:
        key = args.measurement or sorted(f["measurements"])[0]
        g = f["measurements"][key]["libs"]
        wl = g["calibration"][...].astype(np.float64)
        n_rows = int(g["data"].shape[0])
        md = {k: g["metadata"][k][...] for k in ("X", "Y", "X_pos", "Y_pos", "Invalid")}
    sel_all = (md["X"] % args.stride == 0) & (md["Y"] % args.stride == 0) & (md["Invalid"] == 0)
    starts = list(range(0, n_rows, args.slab))
    todo = [s for s in starts if not (blocks / f"block_{s:08d}.npz").is_file()]
    todo = todo[: args.max_blocks] if args.max_blocks else todo
    print(
        f"{h5_path.name} [{key}]: {n_rows:,} shots, {wl.size} px ({wl.min():.1f}-{wl.max():.1f} nm); "
        f"stride {args.stride} -> {int(sel_all.sum()):,} shots; {len(todo)}/{len(starts)} blocks "
        f"to do ({run_info.get('embedding_type')})"
    )

    if not args.plot_only and todo:
        pool = None
        if mode == "tokens":
            le_cfg = yaml.safe_load(open(ROOT / run_info["line_embedding_config"]))
            fit_cfg = {k: le_cfg["line_features"].get(k) for k in FIT_KEYS}
            with h5py.File(ROOT / run_info["line_tokens_path"], "r") as f:
                ld_path = str(f.attrs["line_dict_path"])
            with h5py.File(ld_path, "r") as f:
                centres = f["central_wavelength"][:]
            # Pool first: fork before CUDA is initialised in this process.
            pool = mp.get_context("fork").Pool(args.workers, _init_worker, (wl, centres, fit_cfg))

        clf = load_classifier(run_dir, device, wl, checkpoint=args.checkpoint)
        class_names = clf["class_names"]
        acc = sanity_check(clf, device)
        ref = run_info.get("test_results", {}).get("test/accuracy")
        print(
            f"sanity: test accuracy {acc:.4f} ({args.checkpoint}.ckpt; run_info {ref}, best.ckpt)"
        )
        if args.checkpoint == "best" and ref is not None and abs(acc - float(ref)) > 0.02:
            raise RuntimeError("reloaded model does not reproduce the run's test accuracy")
        json.dump(
            {"class_names": class_names, "sanity_test_accuracy": acc, "mode": clf["mode"]},
            open(out_dir / "classes.json", "w"),
            indent=1,
        )

        t0 = time.time()
        done = 0
        n_todo = int(sum(sel_all[s : s + args.slab].sum() for s in todo))
        for s in todo:
            e = min(s + args.slab, n_rows)
            sel = np.nonzero(sel_all[s:e])[0]
            idx = s + sel
            extra = {}
            if sel.size:
                with h5py.File(h5_path, "r") as f:
                    raw = f["measurements"][key]["libs"]["data"][s:e][sel].astype(np.float32)
                peak = raw.max(axis=1)
                if clf["mode"] == "tokens":
                    X = unit_norm_rows(raw)
                    feats = np.stack(pool.map(_fit_one, list(X), chunksize=4))
                    valid = (feats[..., FEAT_VALID] > 0.5).astype(np.uint8)
                    tokens = np.broadcast_to(clf["static"], (sel.size, *clf["static"].shape)).copy()
                    for fc, tc in FIT_TO_TOKEN.items():
                        tokens[..., tc] = feats[..., fc]
                    probs = classify_tokens(clf["module"], tokens, valid, device)
                    extra["n_valid"] = valid.sum(1).astype(np.int16)
                else:
                    X = patch_input(raw, wl, "counts", clf["input_spec"])
                    probs = classify_spectra(clf["module"], X, device)
            else:
                peak = np.zeros(0, np.float32)
                probs = np.zeros((0, len(class_names)), np.float32)
                if clf["mode"] == "tokens":
                    extra["n_valid"] = np.zeros(0, np.int16)
            np.savez_compressed(
                blocks / f"block_{s:08d}.npz",
                index=idx,
                X=md["X"][idx],
                Y=md["Y"][idx],
                X_pos=md["X_pos"][idx],
                Y_pos=md["Y_pos"][idx],
                peak=peak.astype(np.float32),
                probs=probs.astype(np.float16),
                **extra,
            )
            done += sel.size
            el = time.time() - t0
            eta = el / max(done, 1) * (n_todo - done)
            print(
                f"  block {s:>8d}: {sel.size} shots | {done:,}/{n_todo:,} | "
                f"{el / 60:.1f} min elapsed, ETA {eta / 3600:.2f} h",
                flush=True,
            )
        if pool is not None:
            pool.close()
            pool.join()

    # ── assemble + plot ──
    class_names = json.load(open(out_dir / "classes.json"))["class_names"]
    parts = [
        np.load(blocks / f"block_{s:08d}.npz")
        for s in starts
        if (blocks / f"block_{s:08d}.npz").is_file()
    ]
    fields = ["index", "X", "Y", "X_pos", "Y_pos"]
    fields += [k for k in ("peak", "n_valid") if all(k in p for p in parts)]
    cols = {k: np.concatenate([p[k] for p in parts]) for k in fields}
    probs = np.concatenate([p["probs"] for p in parts]).astype(np.float32)
    ok = np.isfinite(probs).all(1)
    if "n_valid" in cols:
        ok &= cols["n_valid"] > 0
    gate_note = ""
    gated = np.zeros(len(probs), dtype=bool)
    if args.gate_label is not None:
        if "peak" not in cols:
            raise ValueError("these blocks have no raw peak; recompute them to use a gate")
        probs, gated = apply_gate(
            np.nan_to_num(probs), cols["peak"], class_names, args.gate_label, args.gate_peak_counts
        )
        # gated shots always get the gate class (even without a model prediction);
        # an ungated shot whose only probability mass was the gate class has none left
        ok = (ok | gated) & (probs.sum(axis=1) > 0)
        gate_note = f", peak < {args.gate_peak_counts:g} counts -> {args.gate_label}"
    top = np.where(ok, np.nan_to_num(probs).argmax(1), -1)
    df = pd.DataFrame(cols)
    df["predicted"] = [class_names[i] if i >= 0 else None for i in top]
    df["confidence"] = np.where(ok & ~gated, np.nan_to_num(probs).max(1), np.nan)
    second = np.argsort(np.nan_to_num(probs), axis=1)[:, -2]
    df["second"] = [class_names[i] if o and not gt else None for i, o, gt in zip(second, ok, gated)]
    df["gated"] = gated
    df.to_csv(out_dir / "predictions.csv", index=False)

    title = f"{h5_path.stem}: predicted minerals ({run_dir.name}, stride {args.stride}{gate_note})"
    fractions = plot_maps(df, out_dir, title, args.min_frac)
    summary = {
        "h5": str(h5_path),
        "measurement": key,
        "run_dir": str(run_dir),
        "embedding_type": run_info.get("embedding_type"),
        "stride": args.stride,
        "gate": (
            {
                "label": args.gate_label,
                "peak_counts": args.gate_peak_counts,
                "n_gated": int(gated.sum()),
            }
            if args.gate_label
            else None
        ),
        "n_shots": int(len(df)),
        "n_no_prediction": int((~ok).sum()),
        "median_valid_lines": float(np.median(cols["n_valid"])) if "n_valid" in cols else None,
        "median_confidence": float(np.nanmedian(df["confidence"])),
        "complete": len(parts) == len(starts),
        "fractions": fractions,
    }
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "fractions"}, indent=1))
    print("top minerals:", list(fractions.items())[:10])
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
