"""
Select and weight the lines of the CF Saha–Boltzmann solve.

Three things, on the whole measured cache at once (batched ``cf.layer`` solve,
0.5 s per pass on a GPU):

1. **Quality table** (always): per-line Boltzmann self-consistency of the
   deconvolved ("rescued") lines against the fit through the directly fitted
   lines, plus the agreement of deconvolved and Voigt areas on isolated lines.
   Needs no reference values.  → ``line_quality.csv``, ``estimator_agreement.csv``
2. **Rules** (``--rules``): blend / window-r² / bias / scatter filters,
   inverse-variance weights, bias correction and IRLS, each scored against the
   certified values (per-sample medians).  → ``rules.csv``
3. **Training** (``--train``): one weight per dictionary line, shared by all
   spectra, fitted by gradient descent through the solver to the certified
   values with K-fold cross-validation grouped by sample, so the reported
   metrics are out-of-fold.  The result is a line *selection* (the weights go
   to 0 or 1), i.e. a one-time quality calibration of the dictionary, not a
   per-element concentration calibration: the solver stays calibration-free on
   a new spectrum.  → ``line_weights.h5`` (``weight[L]``, fold thetas, out-of-
   fold predictions), ``metrics.yaml``

Usage:
    uv run python scripts/select_cf_lines.py \\
        --tokens external_data/cache/line_tokens_<h>.h5 \\
        --spectra_cache external_data/cache/measured_cache_<h>.h5 \\
        --line_dict external_data/cache/line_dict_<h>.h5 \\
        [--deconv external_data/cache/deconv_<h>.h5 --rescue Fe|all] \\
        --rules --train --out evaluation/cf_lines_<tag>
Then: ``scripts/run_cf_classical.py ... --line_weights <out>/line_weights.h5``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cf.layer import SahaBoltzmannLayer  # noqa: E402
from cf.line_quality import (  # noqa: E402
    CH_AREA, bias_corrected_area, build_masks, estimator_agreement, irls_weights,
    line_consistency, per_line_stats, rule_weights,
)
from cf.tables import build_cf_tables  # noqa: E402
from data.libs_pipeline import load_spectra_cache  # noqa: E402
from train_finetune import build_lod_vector  # noqa: E402

MAJORS = ["Fe", "Cr", "Ni", "Cu", "Mn", "Si", "C", "Mo", "V", "Al", "Ti", "Co"]


def pick_device(want: str) -> str:
    if want != "auto":
        return want
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            return "cuda"
        except RuntimeError:
            print("CUDA busy (exclusive-process mode) - using CPU")
    return "cpu"


class Solver:
    """Batched solve of the whole cache with per-spectrum weights."""

    def __init__(self, layer, tokens, fit_valid, C0, device, chunk, T0, log10_Ne0, log10_Nl0):
        f64 = torch.float64
        self.layer, self.dev, self.chunk = layer, device, chunk
        self.tok = torch.as_tensor(tokens, dtype=f64, device=device)
        self.fv = torch.as_tensor(fit_valid, dtype=f64, device=device)
        self.C0 = None if C0 is None else torch.as_tensor(C0, dtype=f64, device=device)
        self.init = (T0, log10_Ne0, log10_Nl0)
        self.n = self.tok.shape[0]

    def _call(self, j, W):
        f64 = torch.float64
        B = j.numel()
        T0, ne0, nl0 = self.init
        return self.layer(self.tok[j], self.fv[j], W, C0=None if self.C0 is None else self.C0[j],
                          T0=torch.full((B,), T0, dtype=f64, device=self.dev),
                          log10_Ne0=torch.full((B,), ne0, dtype=f64, device=self.dev),
                          log10_Nl0=torch.full((B,), nl0, dtype=f64, device=self.dev))

    def solve(self, W, idx=None, keys=("concentrations", "T", "log10_Ne", "resid", "has_lines", "n_lines_used")):
        idx = np.arange(self.n) if idx is None else np.asarray(idx)
        Wt = W if torch.is_tensor(W) else torch.as_tensor(W, dtype=torch.float64, device=self.dev)
        outs = {k: [] for k in keys}
        with torch.no_grad():
            for s in range(0, idx.size, self.chunk):
                j = torch.as_tensor(idx[s:s + self.chunk], device=self.dev)
                r = self._call(j, Wt[j])
                for k in keys:
                    outs[k].append(r[k].detach().cpu().numpy())
        return {k: np.concatenate(v) for k, v in outs.items()}


def sample_metrics(preds, true, sample_ids, elements, lod):
    """Per-element metrics on per-sample medians, elements above their LOD."""
    out = {}
    P = pd.DataFrame(preds, columns=elements)
    P["s"] = sample_ids
    Tt = pd.DataFrame(true, columns=elements)
    Tt["s"] = sample_ids
    Pm = P.groupby("s").median()
    Tm = Tt.groupby("s").median()
    for e in elements:
        p, tv = Pm[e].to_numpy(), Tm[e].to_numpy()
        m = tv >= lod[e]
        if m.sum() < 3 or np.var(tv[m]) == 0:
            continue
        r2 = 1 - np.sum((tv[m] - p[m]) ** 2) / np.sum((tv[m] - tv[m].mean()) ** 2)
        lr = np.log(p[m] + 1e-7) - np.log(tv[m] + 1e-7)
        out[e] = dict(n=int(m.sum()), r2=float(r2), within_2x=float(np.mean(np.abs(lr) < np.log(2))),
                      log_rmse=float(np.sqrt(np.mean(lr ** 2))), true_median=float(np.median(tv[m])),
                      pred_median=float(np.median(p[m])))
    return out


def summarise(m):
    maj = [m[e] for e in MAJORS if e in m]
    allm = list(m.values())
    return dict(majors_mean_r2=float(np.mean([x["r2"] for x in maj])),
                majors_median_r2=float(np.median([x["r2"] for x in maj])),
                majors_within_2x=float(np.mean([x["within_2x"] for x in maj])),
                all_within_2x=float(np.mean([x["within_2x"] for x in allm])),
                all_log_rmse=float(np.mean([x["log_rmse"] for x in allm])),
                Fe_pred_pct=float(m["Fe"]["pred_median"] * 100) if "Fe" in m else float("nan"),
                Fe_r2=float(m["Fe"]["r2"]) if "Fe" in m else float("nan"))


def fmt(tag, s):
    return (f"{tag:<38s} meanR2 {s['majors_mean_r2']:7.3f}  medR2 {s['majors_median_r2']:6.3f}  "
            f"w2x {s['majors_within_2x']:.3f}  all w2x {s['all_within_2x']:.3f}  "
            f"lrmse {s['all_log_rmse']:.2f} | Fe {s['Fe_pred_pct']:5.1f}% R2 {s['Fe_r2']:6.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--spectra_cache", required=True)
    ap.add_argument("--line_dict", required=True)
    ap.add_argument("--deconv", default=None, help="deconv_*.h5 (scripts/deconvolve_lines.py)")
    ap.add_argument("--rescue", default="Fe", help="elements whose blended lines are rescued: "
                    "comma list or 'all'")
    ap.add_argument("--db", default="external_data/Source/LIBS_data.db")
    ap.add_argument("--element_lod_config", default="config/element_lod.yaml")
    ap.add_argument("--iso_min", type=float, default=0.3)
    ap.add_argument("--r2_min", type=float, default=0.9)
    ap.add_argument("--max_blend", type=float, default=0.8)
    ap.add_argument("--seed_c0", choices=("none", "certified"), default="none",
                    help="C0 seed of the self-absorption init ('certified' only for diagnostics)")
    ap.add_argument("--T0", type=float, default=11000.0)
    ap.add_argument("--log10_Ne0", type=float, default=17.5)
    ap.add_argument("--log10_Nl0", type=float, default=16.0)
    ap.add_argument("--cf_set", action="append", default=[], help="KEY=VALUE solver overrides")
    ap.add_argument("--rules", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--variant", choices=("all", "rescued"), default="all",
                    help="all: every line learns a weight; rescued: direct lines stay at 1")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--l1", type=float, default=0.0, help="L1 on the weights (sparser selection)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out or f"evaluation/cf_lines_{time.strftime('%Y-%m-%d_%H-%M-%S')}")
    out.mkdir(parents=True, exist_ok=True)

    # ── data ────────────────────────────────────────────────────────────
    table, _ = load_spectra_cache(args.spectra_cache)
    elements = [c for c in table.columns if c not in ("sample_type_id", "sample_type_name", "unique_id", "Te", "Ne")]
    tables = build_cf_tables(elements, args.db, args.element_lod_config)
    lod_np, _ = build_lod_vector(elements, args.element_lod_config)
    lod = dict(zip(elements, lod_np))
    with h5py.File(args.tokens, "r") as f:
        TOK = f["tokens"][:].astype(np.float64)
        FV = f["fit_valid"][:].astype(np.float64)
    with h5py.File(args.line_dict, "r") as f:
        iso = f["isolation_score"][:].astype(np.float64)
        sym = np.array([v.decode() if isinstance(v, bytes) else str(v) for v in f["vocab"]["element"][:]])
    n, L, _ = TOK.shape
    E = len(elements)
    if len(table) != n:
        raise SystemExit(f"{args.tokens} has {n} spectra, cache has {len(table)}")
    true = table[elements].to_numpy(dtype=np.float64)
    sid = table["sample_type_id"].astype(str).to_numpy()
    D_AREA = D_BLEND = D_R2 = None
    if args.deconv:
        with h5py.File(args.deconv, "r") as f:
            rows = f["rows"][:]
            if rows.size != n or not np.array_equal(rows, np.arange(n)):
                raise SystemExit(f"{args.deconv} must cover every spectrum of the cache in order")
            D_AREA = f["area"][:].astype(np.float64)
            D_BLEND = f["blend_fraction"][:].astype(np.float64)
            D_R2 = f["window_r2"][:].astype(np.float64)
    rescue = None if args.rescue == "all" else np.isin(sym, args.rescue.split(","))
    TOKA, FVA, masks = build_masks(TOK, FV, iso, D_AREA, D_BLEND, D_R2, rescue,
                                   iso_min=args.iso_min, r2_min=args.r2_min, max_blend=args.max_blend)
    print(f"{n} spectra, {L} lines, {E} elements; direct lines/spectrum median "
          f"{np.median(masks.direct.sum(1)):.0f}, rescued {np.median(masks.rescued.sum(1)):.0f}")

    cfg = dict(min_lines=1, seed_in_closure=False)
    for kv in args.cf_set:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    dev = pick_device(args.device)
    layer = SahaBoltzmannLayer(tables, cfg=cfg).to(dev)
    C0 = true if args.seed_c0 == "certified" else None
    solver = Solver(layer, TOKA, FVA, C0, dev, args.chunk, args.T0, args.log10_Ne0, args.log10_Nl0)
    results = []

    def score(tag, W, extra=None):
        r = solver.solve(W)
        m = sample_metrics(r["concentrations"], true, sid, elements, lod)
        s = summarise(m)
        s.update(tag=tag, T_median=float(np.median(r["T"])),
                 lines_median=float(np.median((np.asarray(W) > 0).sum(1))), **(extra or {}))
        results.append(s)
        print(fmt(tag, s), flush=True)
        return r, m, s

    # ── 1. quality table ────────────────────────────────────────────────
    t0 = time.time()
    W0 = rule_weights(masks, "none")
    r0, m0, s0 = score("baseline: direct lines only", W0)
    print(f"   ({time.time() - t0:.1f} s per pass on {dev})")
    LQ = line_consistency(r0["resid"], masks.rescued, sid, TOKA, sym)
    LQ_direct = line_consistency(r0["resid"], masks.direct, sid, TOKA, sym)
    LQ = LQ.merge(LQ_direct[["line", "bias", "scatter", "n"]].rename(
        columns={"bias": "bias_direct", "scatter": "scatter_direct", "n": "n_direct"}), on="line", how="outer")
    LQ["f_direct"] = masks.direct.mean(0)[LQ["line"].to_numpy()]
    LQ["f_rescued"] = masks.rescued.mean(0)[LQ["line"].to_numpy()]
    if D_BLEND is not None:
        LQ["blend_median"] = [float(np.median(masks.blend[masks.rescued[:, l], l])) if masks.rescued[:, l].any()
                              else np.nan for l in LQ["line"]]
    LQ = LQ.sort_values(["el", "wl"]).reset_index(drop=True)
    LQ.to_csv(out / "line_quality.csv", index=False)
    if D_AREA is not None:
        EA = estimator_agreement(D_AREA, TOK[:, :, CH_AREA], masks.direct, groups=sym)
        EA.to_csv(out / "estimator_agreement.csv", index=False)
        row = EA[EA.group == "all"].iloc[0]
        print(f"deconvolved/Voigt area on {int(row.n)} isolated fits: ratio x{row.ratio:.2f}, sigma {row.sigma:.2f} "
              f"(good: ratio ~1, sigma < 0.4)")
        q = LQ[LQ["n"] >= 30]
        if len(q):
            print(f"rescued lines vs clean fit: {len(q)} lines, |bias| median {q.bias.abs().median():.2f}, "
                  f"scatter median {q.scatter.median():.2f} (ln units)")
    bias_l, scat_l = per_line_stats(LQ[LQ["n"].fillna(0) >= 30], L)

    # ── 2. rules ────────────────────────────────────────────────────────
    if args.rules and D_AREA is not None:
        print("\n=== rules for the rescued lines (direct lines weight 1) ===")
        for mb in (0.8, 0.5, 0.2):
            score(f"blend<={mb} w=1-blend", rule_weights(masks, "blend", max_blend=mb))
        for wr in (0.8, 0.9):
            score(f"blend<=0.8 & window r2>={wr}", rule_weights(masks, "blend", min_window_r2=wr))
        for b in (0.5, 0.3):
            score(f"|bias|<={b}", rule_weights(masks, "filter", bias=bias_l, scatter=scat_l, max_abs_bias=b))
        for sc in (0.5, 0.3):
            score(f"scatter<={sc}", rule_weights(masks, "filter", bias=bias_l, scatter=scat_l, max_scatter=sc))
        score("|bias|<=0.3 & scatter<=0.5", rule_weights(masks, "filter", bias=bias_l, scatter=scat_l,
                                                          max_abs_bias=0.3, max_scatter=0.5))
        for s0 in (0.3, 0.15):
            score(f"inverse variance s0={s0}", rule_weights(masks, "invvar", bias=bias_l, scatter=scat_l, s0=s0))
        tok_keep = solver.tok
        solver.tok = torch.as_tensor(bias_corrected_area(TOKA, masks, bias_l), dtype=torch.float64, device=dev)
        score("bias-corrected areas, blend<=0.8", rule_weights(masks, "blend"))
        score("bias-corrected, scatter<=0.5", rule_weights(masks, "filter", bias=bias_l, scatter=scat_l, max_scatter=0.5))
        solver.tok = tok_keep
        W1 = rule_weights(masks, "blend")
        r1 = solver.solve(W1)
        score("IRLS Cauchy c=0.3 (rescued only)", irls_weights(W1, r1["resid"], 0.3, direct=masks.direct))
        score("IRLS Cauchy c=0.3 (all lines)", irls_weights(W1, r1["resid"], 0.3))
        pd.DataFrame(results).to_csv(out / "rules.csv", index=False)

    # ── 3. trained per-line weights ─────────────────────────────────────
    if args.train:
        if dev == "cpu":
            print("\nWARNING: training on CPU is slow (minutes per fold at full size)")
        print(f"\n=== per-line weights trained on certified values, {args.folds}-fold CV grouped by sample "
              f"(variant '{args.variant}') ===")
        f64 = torch.float64
        lod_t = torch.as_tensor(lod_np, dtype=f64, device=dev)
        true_t = torch.as_tensor(true, dtype=f64, device=dev)
        direct_t = torch.as_tensor(masks.direct, dtype=f64, device=dev)
        resc_t = torch.as_tensor(masks.rescued * (1.0 - masks.blend), dtype=f64, device=dev)
        uniq = np.unique(sid)
        perm = rng.permutation(uniq)
        folds = np.array_split(perm, args.folds)

        def weights_from(theta):
            w = torch.sigmoid(theta)[None, :]
            if args.variant == "rescued":
                return direct_t + resc_t * w
            return (direct_t + resc_t) * w

        def step_loss(theta, idx):
            tot = 0.0
            cnt = 0
            for s in range(0, idx.size, args.chunk):
                j = torch.as_tensor(idx[s:s + args.chunk], device=dev)
                W = weights_from(theta)                      # rebuilt per chunk (own graph)
                r = solver._call(j, W[j])
                pred, tv, hl = r["concentrations"], true_t[j], r["has_lines"]
                above = (tv >= lod_t[None]) & hl
                below = (tv < lod_t[None]) & hl
                lp = torch.log(pred + 1e-7)
                l_ab = ((lp - torch.log(tv + 1e-7)) ** 2)[above].sum()
                l_be = (torch.relu(lp - torch.log(lod_t[None].expand_as(lp))) ** 2)[below].sum()
                loss = (l_ab + l_be) / max(idx.size, 1)
                if args.l1 > 0:
                    loss = loss + args.l1 * torch.sigmoid(theta).sum() * (j.numel() / idx.size)
                loss.backward()
                tot += float(l_ab + l_be)
                cnt += int(above.sum() + below.sum())
            return tot / max(cnt, 1)

        def train(idx):
            theta = torch.full((L,), 2.0, dtype=f64, device=dev, requires_grad=True)
            opt = torch.optim.Adam([theta], lr=args.lr)
            for it in range(args.steps):
                opt.zero_grad()
                lo = step_loss(theta, idx)
                opt.step()
                if it % 50 == 0 or it == args.steps - 1:
                    print(f"      step {it:3d} loss {lo:.4f}", flush=True)
            return theta.detach()

        oof = np.zeros((n, E))
        thetas = []
        for k, test_s in enumerate(folds):
            te = np.where(np.isin(sid, test_s))[0]
            tr = np.where(~np.isin(sid, test_s))[0]
            print(f"   fold {k}: {tr.size} train / {te.size} test spectra")
            th = train(tr)
            thetas.append(th.cpu().numpy())
            oof[te] = solver.solve(weights_from(th), te, keys=("concentrations",))["concentrations"]
        m_oof = sample_metrics(oof, true, sid, elements, lod)
        s_oof = summarise(m_oof)
        s_oof.update(tag=f"trained '{args.variant}' out-of-fold")
        results.append(s_oof)
        print(fmt(s_oof["tag"], s_oof))
        th_mean = np.mean(thetas, 0)
        w_mean = 1 / (1 + np.exp(-th_mean))
        never = (masks.direct | masks.rescued).sum(0) == 0      # no gradient ever: not a decision
        w_mean[never] = 0.0
        score(f"trained '{args.variant}' fold-mean weights (in-sample)", weights_from(
            torch.as_tensor(th_mean, dtype=f64, device=dev)).cpu().numpy())
        print(f"   {int((w_mean > 0.5).sum())} lines kept (weight > 0.5) of {int((~never).sum())} that occur; "
              f"{int(never.sum())} never occur")

        print("\nper element, out-of-fold (per-sample medians, above LOD):")
        print(f"{'el':>3s} {'n':>4s} {'true%':>8s} {'base%':>8s} {'oof%':>8s} {'R2 base':>8s} {'R2 oof':>8s} {'w2x base':>8s} {'w2x oof':>8s}")
        for e in elements:
            if e in m_oof and e in m0:
                a, b = m0[e], m_oof[e]
                print(f"{e:>3s} {b['n']:4d} {b['true_median']*100:8.3f} {a['pred_median']*100:8.3f} {b['pred_median']*100:8.3f} "
                      f"{a['r2']:8.3f} {b['r2']:8.3f} {a['within_2x']:8.2f} {b['within_2x']:8.2f}")
        LQ["weight"] = w_mean[LQ["line"].to_numpy()]
        LQ.to_csv(out / "line_quality.csv", index=False)
        sel = pd.DataFrame(dict(line=np.arange(L), el=sym, wl=TOK[0, :, 0], weight=w_mean,
                                f_direct=masks.direct.mean(0), f_rescued=masks.rescued.mean(0)))
        sel = sel[(sel.f_direct + sel.f_rescued) > 0.05].sort_values(["el", "wl"])
        print("\nlearned weights (lines present in >5 % of spectra):")
        print(sel.round(3).to_string(index=False))
        with h5py.File(out / "line_weights.h5", "w") as f:
            f.create_dataset("weight", data=w_mean)
            f.create_dataset("theta_folds", data=np.asarray(thetas))
            f.create_dataset("oof_predictions", data=oof)
            f.create_dataset("central_wavelength", data=TOK[0, :, 0])
            f.create_dataset("element", data=np.array(sym, dtype="S"))
            f.attrs.update(tokens=args.tokens, line_dict=args.line_dict, deconv=args.deconv or "",
                           rescue=args.rescue, variant=args.variant, max_blend=args.max_blend,
                           iso_min=args.iso_min, r2_min=args.r2_min, folds=args.folds,
                           steps=args.steps, lr=args.lr, elements=json.dumps(elements))
        print(f"\nweights written: {out / 'line_weights.h5'}")

    with open(out / "metrics.yaml", "w") as fh:
        yaml.safe_dump({"runs": results, "args": vars(args)}, fh, sort_keys=False)
    print(f"\noutputs in {out}")


if __name__ == "__main__":
    main()
