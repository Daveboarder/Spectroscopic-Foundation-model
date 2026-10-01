"""
Synthetic-to-real transfer screen against a TIMA phase map (no deep model).

Answers "does input preprocessing X let a classifier trained on synthetic spectra name the
minerals of a real LIBS map?" in minutes on the CPU. A logistic regression is trained on
synthetic spectra of a data config (0.5 nm bin means of asinh(model input)) and applied to
real rock shots of the map, labelled by the registered TIMA phase map.

Sub-commands
  register          register the TIMA phase PNG to the LIBS raster (rock outline, IoU search)
                    from the raw peaks stored in a predict_mineral_map.py block directory;
                    writes <out>/tgrid.npy (TIMA class per raster cell) + registration.json
  measure_nuisance  per-channel noise (robust first-difference MAD), red-channel continuum,
                    zero fractions and saturation of real rock shots (counts)
  screen            linear transfer arms, e.g.
                      --arm A=config/libs_data_minerals_avantes.yaml:none
                      --arm B=config/libs_data_minerals_avantes_fw.yaml:canonical
                      --arm B2=config/libs_data_minerals_avantes_fw.yaml:canonical:median
                    label sets oracle3 / plausible10 / all20 / all63, standardisation on the
                    synthetic source only (srcstd) or per domain (domstd: real statistics from
                    the other half of the slabs), scored on TIMA grain-interior shots (label
                    constant within +-2 cells) and on all labelled rock shots. Reports macro
                    recall (plagioclase = Albite + Oligoclase, white mica = Muscovite +
                    Lepidolite), sink share, epoxid share, AMI, a real-label probe
                    (information check), the real-vs-synthetic domain accuracy and a paired
                    bootstrap over slabs against the first arm.
  port_check        arm 'as-is' on the 8 slabs / split of the scratch E0/E1 screen; must give
                    oracle-3 interior balanced recall ~0.661 and domain accuracy ~0.975.
  evaluate_runs     trained spectral_patch classification runs on the same shots, e.g.
                      --group A=runs/r1,runs/r2 --group B=runs/r3,runs/r4 --checkpoint last
                    per run: interior macro recall with 'epoxid' masked (all classes), epoxid
                    share of all rock shots before masking, sink share, AMI and the synthetic
                    test accuracy of the loaded checkpoint; per group the seed mean, and a
                    paired slab bootstrap of each group's seed-mean against the first group
                    with the G2 gate (>= +0.10 with CI above 0, epoxid <= 0.2, sink <= 0.3,
                    synthetic accuracy >= 0.9). Runs with boundary-mixture classes ("A + B",
                    generation.mixtures) also get a lenient interior recall (the true mineral is
                    the class or a component of the predicted mixture), the interior mixture
                    share, and boundary scores on shots whose 3x3 TIMA neighbourhood holds
                    exactly two of albite / quartz / muscovite: mixture share, pair accuracy
                    (predicted mixture = that pair) and component accuracy (>= one of them).

Rock = raw peak >= 3000 counts and not Invalid. Raster rows = (NY - 1 - Y), cols = X
(Mar1A: 994 x 961 cells, 80 um). Outputs JSON next to the requested --out path.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.canonical import channel_bounds, patch_input, resolve_spec  # noqa: E402

TIMA_NAMES = [
    "albite", "quartz", "muscovite", "spessartine", "orthoclase", "biotite",
    "unclassified", "holes", "background",
]  # fmt: skip
MAR1A_COLOURS = {  # RGB of the Marsikov core legend (MArsikov_core-legend.png)
    "albite": (71, 213, 213), "quartz": (85, 0, 255), "muscovite": (251, 119, 255),
    "spessartine": (190, 95, 41), "orthoclase": (255, 32, 103), "biotite": (255, 170, 0),
    "unclassified": (0, 0, 0), "holes": (128, 128, 128), "background": (255, 255, 255),
}  # fmt: skip
TRUTH = {0: "Albite", 1: "Quartz", 2: "Muscovite"}  # TIMA index -> synthetic class
GROUP = {"Oligoclase (plagioclase An10-30)": "Albite", "Lepidolite": "Muscovite"}
LABEL_SETS = {
    "oracle3": ["Albite", "Quartz", "Muscovite"],
    "plausible10": [
        "Albite", "Quartz", "Muscovite", "Orthoclase", "Oligoclase (plagioclase An10-30)",
        "Biotite (Fe-rich, annite)", "Spessartine", "Beryl", "Schorl", "Lepidolite",
    ],
    "all20": [
        "Albite", "Quartz", "Muscovite", "Orthoclase", "Oligoclase (plagioclase An10-30)",
        "Biotite (Fe-rich, annite)", "Spessartine", "Beryl", "Schorl", "Calcite", "Fluorapatite",
        "Kaolinite", "Lepidolite", "Elbaite (Li-tourmaline)", "Cassiterite", "Columbite-(Fe)",
        "Hematite", "Magnetite", "Zircon", "Monazite-(Ce)",
    ],
    "all63": None,  # every class of the data config (incl. measured extra classes)
}  # fmt: skip
ROCK_PEAK = 3000.0


# ─────────────────────────────────────────────────────────────────────────────
# TIMA registration
# ─────────────────────────────────────────────────────────────────────────────
def tima_classes(png: str, colours: dict, step: int = 8) -> np.ndarray:
    """TIMA phase PNG -> [H, W] index into TIMA_NAMES (nearest legend colour, 1/step scale)."""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(png).convert("RGB")
    small = np.asarray(im.resize((im.size[0] // step, im.size[1] // step), Image.NEAREST))
    ref = np.array([colours[n] for n in TIMA_NAMES], dtype=int)
    d = ((small.astype(int)[:, :, None, :] - ref[None, None]) ** 2).sum(-1)
    return d.argmin(-1)


def load_block_peaks(block_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, Y, P = [], [], []
    for p in sorted(Path(block_dir).glob("block_*.npz")):
        z = np.load(p)
        if "peak" not in z:
            raise ValueError(f"{p} has no raw peak (recompute the map blocks)")
        X.append(z["X"])
        Y.append(z["Y"])
        P.append(z["peak"])
    return np.concatenate(X), np.concatenate(Y), np.concatenate(P)


def register(tima: np.ndarray, X, Y, peak) -> tuple[np.ndarray, dict]:
    """Rock-outline registration (bounding boxes + IoU grid search over scale and shift)."""
    ny, nx = int(Y.max()) + 1, int(X.max()) + 1
    grid_peak = np.full((ny, nx), np.nan)
    grid_peak[ny - 1 - Y, X] = peak
    l_rock = grid_peak >= ROCK_PEAK
    H, W = tima.shape
    t_rock = tima != TIMA_NAMES.index("background")
    rr, cc = np.nonzero(t_rock)
    T0, T1, C0, C1 = rr.min(), rr.max(), cc.min(), cc.max()
    r, c = np.nonzero(l_rock)
    R0, R1, K0, K1 = r.min(), r.max(), c.min(), c.max()
    gr, gc = np.mgrid[0:ny, 0:nx]
    best = None
    for sr in np.linspace(0.97, 1.03, 7):
        for sc in np.linspace(0.97, 1.03, 7):
            for dr in range(-6, 7, 2):
                for dc in range(-6, 7, 2):
                    tr = T0 + (gr - R0 - dr) * sr * (T1 - T0) / (R1 - R0)
                    tc = C0 + (gc - K0 - dc) * sc * (C1 - C0) / (K1 - K0)
                    ok = (tr >= 0) & (tr < H) & (tc >= 0) & (tc < W)
                    tm = np.zeros((ny, nx), bool)
                    tm[ok] = t_rock[tr[ok].astype(int), tc[ok].astype(int)]
                    iou = (tm & l_rock).sum() / (tm | l_rock).sum()
                    if best is None or iou > best[0]:
                        best = (iou, sr, sc, dr, dc, tr, tc, ok)
    iou, sr, sc, dr, dc, tr, tc, ok = best
    tgrid = np.full((ny, nx), TIMA_NAMES.index("background"))
    tgrid[ok] = tima[tr[ok].astype(int), tc[ok].astype(int)]
    info = {"iou": round(float(iou), 4), "scale": [float(sr), float(sc)], "shift": [dr, dc],
            "grid": [ny, nx], "rows": "NY - 1 - Y", "cols": "X", "names": TIMA_NAMES}  # fmt: skip
    return tgrid, info


def interior_mask(tgrid: np.ndarray, rows: np.ndarray, cols: np.ndarray, r: int = 2) -> np.ndarray:
    """True where the TIMA label is constant within +-r cells."""
    H, W = tgrid.shape
    ok = np.ones(len(rows), bool)
    for dr in range(-r, r + 1):
        for dc in range(-r, r + 1):
            ok &= (
                tgrid[np.clip(rows + dr, 0, H - 1), np.clip(cols + dc, 0, W - 1)]
                == tgrid[rows, cols]
            )
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
def load_real(h5: str, n_slabs: int, lo: float = 0.1, hi: float = 0.9, slab: int = 1024) -> dict:
    """Evenly spaced chunk-aligned slabs of ``slab`` consecutive shots (raw counts)."""
    with h5py.File(h5, "r") as f:
        g = f["measurements"][sorted(f["measurements"])[0]]["libs"]
        d = g["data"]
        n = d.shape[0]
        starts = (np.linspace(lo, hi, n_slabs) * n).astype(int) // slab * slab
        raw = np.concatenate([d[s : s + slab] for s in starts]).astype(np.float32)
        idx = np.concatenate([np.arange(s, s + slab) for s in starts])
        md = {k: g["metadata"][k][...] for k in ("X", "Y", "Invalid")}
        wl = g["calibration"][...].astype(np.float64)
    return {"raw": raw, "idx": idx, "slab": np.repeat(np.arange(len(starts)), slab),
            "X": md["X"][idx], "Y": md["Y"][idx], "invalid": md["Invalid"][idx],
            "ny": int(md["Y"].max()) + 1, "wl": wl}  # fmt: skip


def load_synthetic(data_cfg: str) -> dict:
    from data.libs_pipeline import build_dataset_from_config

    cfg = yaml.safe_load(open(ROOT / data_cfg))
    cfg.setdefault("generation", {})["verbose"] = False
    ds = build_dataset_from_config(cfg)
    return {"x": np.asarray(ds.spectra, np.float32), "units": getattr(ds, "units", "unit_norm"),
            "names": ds.sample_table["sample_type_name"].astype(str).to_numpy(),
            "wl": np.asarray(ds.wavelength, np.float64), "cfg": cfg}  # fmt: skip


def binner(wl: np.ndarray, width: float = 0.5):
    bins = np.arange(np.floor(wl.min()), np.ceil(wl.max()) + width, width)
    bid = np.clip(np.digitize(wl, bins) - 1, 0, len(bins) - 2)
    cnt = np.bincount(bid, minlength=len(bins) - 1)
    good = cnt > 0
    M = np.zeros((wl.size, len(bins) - 1), np.float32)  # mean pooling as a matrix product
    M[np.arange(wl.size), bid] = 1.0 / np.maximum(cnt[bid], 1)
    return M[:, good]


def features(x, wl, units, spec, M, input_scale) -> np.ndarray:
    out = []
    for s in range(0, len(x), 2048):
        v = patch_input(x[s : s + 2048], wl, units, spec)[:, : wl.size]
        out.append(np.arcsinh(v / input_scale) @ M)
    return np.concatenate(out).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def components(name: str) -> frozenset:
    """Grouped mineral(s) of a class: 'A + B' (boundary mixture) -> {A, B}."""
    return frozenset(GROUP.get(p, p) for p in str(name).split(" + "))


def boundary_score(pred_names: np.ndarray, pairs: list) -> dict:
    """Shots on a two-mineral boundary: mixture share, exact-pair and any-component accuracy."""
    comp = [components(c) for c in pred_names]
    is_mix = np.array([" + " in str(c) for c in pred_names])
    return {
        "n": int(len(pairs)),
        "mixture_share": round(float(is_mix.mean()), 4) if len(pairs) else 0.0,
        "pair_acc": round(
            float(np.mean([m and c == p for m, c, p in zip(is_mix, comp, pairs)])), 4
        ),
        "component_acc": round(float(np.mean([bool(c & p) for c, p in zip(comp, pairs)])), 4),
    }


def score(pred_names: np.ndarray, t: np.ndarray, slab: np.ndarray) -> dict:
    from sklearn.metrics import adjusted_mutual_info_score

    truth = np.array([TRUTH[v] for v in t])
    pg = np.array([GROUP.get(c, c) for c in pred_names])
    rec = {c: float((pg[truth == c] == c).mean()) for c in TRUTH.values() if (truth == c).any()}
    comp = [components(c) for c in pred_names]
    rec_len = {c: float(np.mean([c in comp[k] for k in np.flatnonzero(truth == c)]))
               for c in TRUTH.values() if (truth == c).any()}  # fmt: skip
    vals, cnt = np.unique(pg, return_counts=True)
    wrong = [(c, k) for c, k in zip(vals, cnt) if c not in TRUTH.values()]
    sink = max(wrong, key=lambda ck: ck[1]) if wrong else ("-", 0)
    return {
        "macro_recall": round(float(np.mean(list(rec.values()))), 4),
        "recall": {c: round(v, 3) for c, v in rec.items()},
        "named_acc": round(float((pg == truth).mean()), 4),
        "sink": [str(sink[0]), round(float(sink[1] / len(pg)), 3)],
        "epoxid_share": round(float((pred_names == "epoxid").mean()), 3),
        "AMI": round(float(adjusted_mutual_info_score(truth, pg)), 4),
        "n": int(len(pg)),
        "macro_recall_lenient": round(float(np.mean(list(rec_len.values()))), 4),
        "recall_lenient": {c: round(v, 3) for c, v in rec_len.items()},
        "mixture_share": round(float(np.mean([" + " in str(c) for c in pred_names])), 4),
    }


def slab_bootstrap(pa, pb, t, slab, n=1000, seed=0) -> list[float]:
    """95 % CI of macro_recall(b) - macro_recall(a), resampling slabs with replacement."""
    rng = np.random.default_rng(seed)
    ids = np.unique(slab)

    def macro(pred, sel):
        truth = np.array([TRUTH[v] for v in t[sel]])
        pg = np.array([GROUP.get(c, c) for c in pred[sel]])
        return np.mean([(pg[truth == c] == c).mean() for c in TRUTH.values() if (truth == c).any()])

    diffs = []
    for _ in range(n):
        pick = rng.choice(ids, size=len(ids), replace=True)
        sel = np.concatenate([np.flatnonzero(slab == i) for i in pick])
        diffs.append(macro(pb, sel) - macro(pa, sel))
    return [round(float(v), 4) for v in np.percentile(diffs, [2.5, 50, 97.5])]


# ─────────────────────────────────────────────────────────────────────────────
def cmd_register(a):
    X, Y, P = load_block_peaks(Path(a.blocks))
    tima = tima_classes(a.tima_png, MAR1A_COLOURS)
    tgrid, info = register(tima, X, Y, P)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "tgrid.npy", tgrid)
    info.update(tima_png=str(a.tima_png), blocks=str(a.blocks))
    json.dump(info, open(out / "registration.json", "w"), indent=1)
    print(json.dumps(info))


def rock_and_labels(R: dict, tgrid: np.ndarray):
    peak = R["raw"].max(1)
    rows, cols = R["ny"] - 1 - R["Y"], R["X"]
    t = tgrid[rows, cols]
    rock = (peak >= ROCK_PEAK) & (R["invalid"] == 0)
    lab = rock & np.isin(t, list(TRUTH))
    return rock, lab, t, interior_mask(tgrid, rows, cols, 2)


def cmd_measure(a):
    R = load_real(a.h5, a.n_slabs)
    peak = R["raw"].max(1)
    rock = (peak >= ROCK_PEAK) & (R["invalid"] == 0)
    x = R["raw"][rock]
    res = {
        "n_rock": int(rock.sum()),
        "saturated_fraction": round(float((peak[rock] >= 64000).mean()), 3),
    }
    for k, (lo, hi) in enumerate(channel_bounds(R["wl"])):
        seg = x[:, lo:hi]
        d = np.diff(seg, axis=1)
        mad = (
            1.4826 * np.median(np.abs(d - np.median(d, axis=1, keepdims=True)), axis=1) / np.sqrt(2)
        )
        res[f"channel{k}"] = {
            "nm": [round(float(R["wl"][lo]), 2), round(float(R["wl"][hi - 1]), 2)],
            "noise_sigma_counts_median": round(float(np.median(mad)), 2),
            "p5_level_counts_median": round(float(np.median(np.percentile(seg, 5, axis=1))), 1),
            "zero_fraction": round(float((seg == 0).mean()), 4),
        }
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


def cmd_screen(a, port_check: bool = False):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    t0 = time.time()
    tgrid = np.load(Path(a.registration) / "tgrid.npy")
    R = load_real(a.h5, 8 if port_check else a.n_slabs)
    rock, lab, t, interior = rock_and_labels(R, tgrid)
    print(f"real: {len(R['raw'])} shots, rock {int(rock.sum())}, labelled {int(lab.sum())}, "
          f"interior {int((lab & interior).sum())} ({time.time() - t0:.0f} s)", flush=True)  # fmt: skip
    arms = [s.split("=", 1) for s in a.arm]
    if port_check:
        arms = [("asis", "config/libs_data_minerals_avantes.yaml:none")]
    half = R["slab"] % 2  # 2-fold over slabs for per-domain statistics (never the scored slabs)
    if port_check:
        half = (R["slab"] >= 4).astype(int)
    results, preds = {"n_rock": int(rock.sum()), "n_labelled": int(lab.sum())}, {}
    for name, spec_s in arms:
        parts = spec_s.split(":")
        data_cfg, mode = parts[0], parts[1] if len(parts) > 1 else "none"
        syn = load_synthetic(data_cfg)
        spec = resolve_spec({"patch": {"preprocess": mode}}, syn["cfg"])
        if len(parts) > 2:
            spec["canonical"]["baseline_method"] = parts[2]
        if mode == "canonical":
            spec["canonical"]["noise_sigma_counts"] = (
                a.noise_sigma or spec["canonical"]["noise_sigma_counts"]
            )
        scale = 0.01 if mode == "none" else 1.0
        M = binner(R["wl"])
        Fr = features(R["raw"], R["wl"], "counts", spec, M, scale)
        Fs = features(syn["x"], syn["wl"], syn["units"], spec, binner(syn["wl"]), scale)
        arm = {"data_cfg": data_cfg, "preprocess": mode, "spec": spec}
        # information check: real-label probe, leave-one-slab-out (interior shots)
        sel = lab & interior
        yl = t[sel]
        pr = np.empty_like(yl)
        for s_id in np.unique(R["slab"][sel]):
            tr, te = R["slab"][sel] != s_id, R["slab"][sel] == s_id
            if len(np.unique(yl[tr])) < 2:
                pr[te] = np.bincount(yl[tr]).argmax()
                continue
            sc = StandardScaler().fit(Fr[sel][tr])
            clf = LogisticRegression(C=0.1, max_iter=3000).fit(sc.transform(Fr[sel][tr]), yl[tr])
            pr[te] = clf.predict(sc.transform(Fr[sel][te]))
        arm["real_label_probe_interior_balanced"] = round(float(balanced_accuracy_score(yl, pr)), 4)
        for lset, classes in LABEL_SETS.items():
            classes = classes or sorted(set(syn["names"]))
            m = np.isin(syn["names"], classes)
            if m.sum() == 0:
                continue
            y = syn["names"][m]
            sc_s = StandardScaler().fit(Fs[m])
            clf = LogisticRegression(C=0.1, max_iter=3000).fit(sc_s.transform(Fs[m]), y)
            for std in ("srcstd", "domstd"):
                pred = np.empty(len(Fr), dtype=object)
                for h in (0, 1):
                    fit_on, apply_to = (half != h) & rock, half == h
                    if port_check and h == 0:
                        continue  # E1: statistics from slabs 0-3, scored on slabs 4-7
                    sc_r = sc_s if std == "srcstd" else StandardScaler().fit(Fr[fit_on])
                    pred[apply_to] = clf.predict(sc_r.transform(Fr[apply_to]))
                scored = lab & (half == 1) if port_check else lab
                key = f"{lset}|{std}"
                arm[key] = {
                    "interior_r2": score(
                        pred[scored & interior].astype(str), t[scored & interior], None
                    ),
                    "all": score(pred[scored].astype(str), t[scored], None),
                }
                preds[(name, key)] = pred
        # domain separability (rock vs the all20 synthetic classes, balanced, 5-fold)
        m20 = np.isin(syn["names"], LABEL_SETS["all20"])
        Xd = np.vstack([Fr[rock], Fs[m20]])
        yd = np.r_[np.zeros(int(rock.sum())), np.ones(int(m20.sum()))]
        arm["domain_acc"] = round(float(cross_val_score(
            LogisticRegression(C=0.1, max_iter=3000), StandardScaler().fit_transform(Xd), yd, cv=5,
            scoring="balanced_accuracy").mean()), 4)  # fmt: skip
        results[name] = arm
        brief = {
            k: v["interior_r2"]["macro_recall"]
            for k, v in arm.items()
            if isinstance(v, dict) and "interior_r2" in v
        }
        print(f"[{name}] {mode}: probe {arm['real_label_probe_interior_balanced']}, domain "
              f"{arm['domain_acc']}, interior macro recall {brief} ({time.time() - t0:.0f} s)", flush=True)  # fmt: skip
    if not port_check and len(arms) > 1:
        ref = arms[0][0]
        sel = lab & interior
        boot = {}
        for (name, key), pred in preds.items():
            if name == ref:
                continue
            boot[f"{name}-{ref}|{key}"] = slab_bootstrap(
                preds[(ref, key)][sel].astype(str), pred[sel].astype(str), t[sel], R["slab"][sel]
            )
        results["paired_bootstrap_interior_macro_recall"] = boot
    json.dump(results, open(a.out, "w"), indent=1, default=str)
    print(f"-> {a.out} ({time.time() - t0:.0f} s)")
    return results


def cmd_evaluate_runs(a):
    import torch

    import scripts.predict_mineral_map as pmm

    t0 = time.time()
    tgrid = np.load(Path(a.registration) / "tgrid.npy")
    R = load_real(a.h5, a.n_slabs)
    rock, lab, t, interior = rock_and_labels(R, tgrid)
    sel = (lab & interior)[rock]  # scored shots within the rock shots
    raw_rock = R["raw"][rock]
    # two-mineral boundaries: 3x3 TIMA neighbourhood = exactly two of albite / quartz / muscovite
    rows, cols = R["ny"] - 1 - R["Y"], R["X"]
    H, W = tgrid.shape
    neigh = np.stack([tgrid[np.clip(rows + dr, 0, H - 1), np.clip(cols + dc, 0, W - 1)]
                      for dr in (-1, 0, 1) for dc in (-1, 0, 1)], 1)  # fmt: skip
    srt = np.sort(neigh, 1)
    n_distinct = (srt[:, 1:] != srt[:, :-1]).sum(1) + 1
    bnd = rock & lab & np.isin(neigh, list(TRUTH)).all(1) & (n_distinct == 2)
    sel_b = bnd[rock]
    pairs_b = [frozenset(TRUTH[v] for v in np.unique(r)) for r in neigh[bnd]]
    groups = [g.split("=", 1) for g in a.group]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results, preds = {"checkpoint": a.checkpoint, "n_rock": int(rock.sum()), "groups": {}}, {}
    for gname, runs in groups:
        rows = []
        for run in [r for r in runs.split(",") if r]:
            clf = pmm.load_classifier(Path(run), device, R["wl"], checkpoint=a.checkpoint)
            if clf["mode"] != "patch":
                raise ValueError(f"{run}: evaluate_runs handles spectral_patch runs only")
            names = np.array(clf["class_names"])
            P = np.concatenate([
                pmm.classify_spectra(clf["module"], patch_input(raw_rock[s : s + 2048], R["wl"], "counts", clf["input_spec"]), device)
                for s in range(0, len(raw_rock), 2048)
            ])  # fmt: skip
            ep = int(np.flatnonzero(names == "epoxid")[0]) if "epoxid" in names else None
            unmasked = P.argmax(1)
            if ep is not None:
                P[:, ep] = -np.inf
            pred = names[P.argmax(1)]
            row = score(pred[sel], t[rock][sel], None)
            row["boundary"] = boundary_score(pred[sel_b], pairs_b)
            row["epoxid_share_rock_unmasked"] = (
                round(float((unmasked == ep).mean()), 4) if ep is not None else 0.0
            )
            row["synthetic_test_accuracy"] = round(pmm.sanity_check(clf, device), 4)
            row["run"] = run
            rows.append(row)
            preds.setdefault(gname, []).append(pred[sel])
            print(f"[{gname}] {run}: macro {row['macro_recall']:.3f} {row['recall']} epoxid(unmasked) "
                  f"{row['epoxid_share_rock_unmasked']} sink {row['sink']} syn {row['synthetic_test_accuracy']} "
                  f"| lenient {row['macro_recall_lenient']:.3f} {row['recall_lenient']} mixture share "
                  f"interior {row['mixture_share']:.3f} / boundary {row['boundary']['mixture_share']:.3f} "
                  f"(n {row['boundary']['n']}), boundary pair {row['boundary']['pair_acc']:.3f} "
                  f"component {row['boundary']['component_acc']:.3f} ({time.time() - t0:.0f} s)", flush=True)  # fmt: skip
        mean = {k: round(float(np.mean([r[k] for r in rows])), 4)
                for k in ("macro_recall", "epoxid_share_rock_unmasked", "synthetic_test_accuracy", "AMI")}  # fmt: skip
        mean["sink_share"] = round(float(np.mean([r["sink"][1] for r in rows])), 4)
        mean["macro_recall_lenient"] = round(
            float(np.mean([r["macro_recall_lenient"] for r in rows])), 4
        )
        mean["interior_mixture_share"] = round(
            float(np.mean([r["mixture_share"] for r in rows])), 4
        )
        for k in ("mixture_share", "pair_acc", "component_acc"):
            mean["boundary_" + k] = round(float(np.mean([r["boundary"][k] for r in rows])), 4)
        results["groups"][gname] = {"runs": rows, "mean": mean}
    # paired slab bootstrap of seed-mean macro recall against the first group
    ref = groups[0][0]
    slab = R["slab"][rock][sel]
    truth = np.array([TRUTH[v] for v in t[rock][sel]])
    rng = np.random.default_rng(0)
    ids = np.unique(slab)
    members = [np.flatnonzero(slab == i) for i in ids]

    def macro(p, idx):
        pg = np.array([GROUP.get(c, c) for c in p[idx]])
        tr = truth[idx]
        return np.mean([(pg[tr == c] == c).mean() for c in TRUTH.values() if (tr == c).any()])

    draws = [
        np.concatenate([members[j] for j in rng.integers(0, len(ids), len(ids))])
        for _ in range(1000)
    ]
    for gname, _ in groups[1:]:
        diffs = [np.mean([macro(p, d) for p in preds[gname]]) - np.mean([macro(p, d) for p in preds[ref]])
                 for d in draws]  # fmt: skip
        ci = [round(float(v), 4) for v in np.percentile(diffs, [2.5, 50, 97.5])]
        m = results["groups"][gname]["mean"]
        gate = (ci[1] >= 0.10 and ci[0] > 0 and m["epoxid_share_rock_unmasked"] <= 0.2
                and m["sink_share"] <= 0.3 and m["synthetic_test_accuracy"] >= 0.9)  # fmt: skip
        results["groups"][gname]["vs_" + ref] = {"macro_recall_diff_ci": ci, "G2_pass": bool(gate)}
        print(f"{gname} - {ref}: macro recall difference {ci} -> G2 {'PASS' if gate else 'fail'}")
    out = a.out or str(ROOT / "Outputs" / "tima_transfer_evaluate_runs.json")
    json.dump(results, open(out, "w"), indent=1, default=str)
    print(f"-> {out} ({time.time() - t0:.0f} s)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register")
    r.add_argument("--blocks", required=True, help="predict_mineral_map block dir with raw peaks")
    r.add_argument("--tima_png", required=True)
    r.add_argument("--out", required=True, help="output dir (tgrid.npy, registration.json)")
    e = sub.add_parser("evaluate_runs")
    e.add_argument("--h5", required=True)
    e.add_argument("--n_slabs", type=int, default=24)
    e.add_argument("--registration", required=True)
    e.add_argument("--group", action="append", required=True, help="NAME=run_dir[,run_dir...]")
    e.add_argument("--checkpoint", choices=("best", "last"), default="last")
    e.add_argument("--out", default=None)
    for name in ("measure_nuisance", "screen", "port_check"):
        p = sub.add_parser(name)
        p.add_argument("--h5", required=True)
        p.add_argument("--n_slabs", type=int, default=24)
        p.add_argument("--out", default=None)
        if name != "measure_nuisance":
            p.add_argument("--registration", required=True, help="dir with tgrid.npy")
            p.add_argument(
                "--arm", action="append", default=[], help="NAME=DATA_CFG:PREPROCESS[:BASELINE]"
            )
            p.add_argument("--noise_sigma", type=float, nargs="+", default=None,
                           help="per-channel noise sigma (counts) for canonical arms")  # fmt: skip
    a = ap.parse_args()
    if a.cmd == "register":
        cmd_register(a)
    elif a.cmd == "measure_nuisance":
        cmd_measure(a)
    elif a.cmd == "evaluate_runs":
        cmd_evaluate_runs(a)
    else:
        a.out = a.out or str(ROOT / "Outputs" / f"tima_transfer_{a.cmd}.json")
        cmd_screen(a, port_check=a.cmd == "port_check")


if __name__ == "__main__":
    main()
