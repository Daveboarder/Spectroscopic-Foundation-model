"""
Classical (zero-parameter) calibration-free quantification over a token cache.

Runs ``cf.solver_np.saha_boltzmann_solve_np`` with ``cf.classical.classical_weights``
on every requested spectrum of a ``line_tokens_*.h5`` cache, compares the
mass fractions with the ``sample_table`` of the matching spectra cache, and
writes:

    <out>/per_spectrum.csv   unique_id, T, log10_Ne, n_lines, <El>, <El>_censored ...
    <out>/per_element.yaml   mae, log_rmse, within_2x, r2, n per element (+ per-sample medians)

Usage:
    uv run python scripts/run_cf_classical.py \\
        --tokens external_data/cache/line_tokens_<hash>.h5 \\
        --spectra_cache external_data/cache/measured_cache_<hash>.h5 \\
        --line_dict external_data/cache/line_dict_<hash>.h5 \\
        --line_list cf_oes54 --sample "PURE KFE" --indices 20

``--line_list cf_oes54`` restricts the solver to the 54 curated CF spark-OES
lines (matched by element, stage and wavelength), ``cf_minerals`` to the
REE-mineral LIBS-in-air list ``cf/data/cf_mineral_lines.tsv``; ``all`` uses every line
that passes the fit-quality gate.  Values below ``config/element_lod.yaml``
are reported as censored.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cf.classical import (  # noqa: E402
    CF_MINERAL_TSV,
    CF_OES_54_TSV,
    classical_weights,
    load_cf_oes_lines,
    select_cf_oes_lines,
)
from cf.solver_np import saha_boltzmann_solve_np  # noqa: E402
from cf.tables import build_cf_tables  # noqa: E402
from data.libs_pipeline import _META_COLS, load_spectra_cache  # noqa: E402

_CH_WL, _CH_Z, _CH_ION = 0, 7, 8


def _element_columns(table: pd.DataFrame) -> list[str]:
    return [c for c in table.columns if c not in _META_COLS and not c.startswith("gamma_stark")
            and c not in ("plasma_model", "Te1", "Ne1", "Te2", "Ne2", "l_inner", "l_outer", "N1", "N2")]


def _metrics(pred: np.ndarray, true: np.ndarray, lod: float, eps: float = 1e-7) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    m = true >= lod
    out: dict[str, float] = {"n": int(true.size), "n_above_lod": int(m.sum()),
                             "mae": float(np.mean(np.abs(pred - true)))}
    if m.sum() >= 1:
        lr = np.log(pred[m] + eps) - np.log(true[m] + eps)
        out["log_rmse"] = float(np.sqrt(np.mean(lr ** 2)))
        out["within_2x"] = float(np.mean(np.abs(lr) < np.log(2.0)))
    if m.sum() >= 2 and np.var(true[m]) > 0:
        ss_res = float(np.sum((true[m] - pred[m]) ** 2))
        ss_tot = float(np.sum((true[m] - true[m].mean()) ** 2))
        out["r2"] = float(1.0 - ss_res / ss_tot)
    return out


def _resolve_indices(spec: str, table: pd.DataFrame, sample: str | None, cache_dir: Path,
                     n_total: int) -> np.ndarray:
    idx = np.arange(n_total)
    if sample:
        names = table["sample_type_name"].astype(str).to_numpy()
        ids = table["sample_type_id"].astype(str).to_numpy()
        hit = np.array([sample.lower() in (a + " " + b).lower() for a, b in zip(names, ids)])
        idx = idx[hit]
        if idx.size == 0:
            raise SystemExit(f"no spectra match --sample {sample!r}")
    if spec == "all":
        return idx
    if spec == "test":
        cands = sorted(cache_dir.glob("splits_*.json"))
        for c in cands:
            s = json.load(open(c))
            if sum(len(v) for v in s.values()) == n_total:
                test = np.asarray(s["test"], dtype=np.int64)
                return np.intersect1d(idx, test)
        raise SystemExit(f"no splits_*.json with {n_total} rows in {cache_dir}")
    n = int(spec)
    return idx[:n]


def main() -> None:
    ap = argparse.ArgumentParser(description="Classical CF quantification over a token cache.")
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--spectra_cache", required=True)
    ap.add_argument("--line_dict", default=None, help="line_dict_*.h5 (isolation_score / forced)")
    ap.add_argument("--db", default="external_data/Source/LIBS_data.db")
    ap.add_argument("--element_lod_config", default="config/element_lod.yaml")
    ap.add_argument("--indices", default="all", help="all | test | <n first matching>")
    ap.add_argument("--line_list", choices=("all", "cf_oes54", "cf_minerals"), default="all")
    ap.add_argument("--deconv", default=None,
                    help="deconv_*.h5 from scripts/deconvolve_lines.py: use the deconvolved "
                         "areas of blended lines instead of their single-Voigt fit")
    ap.add_argument("--deconv_max_blend", type=float, default=0.8,
                    help="only take a deconvolved area when the line owns more than "
                         "1 - this fraction of its own peak")
    ap.add_argument("--deconv_only_invalid", action="store_true",
                    help="keep every good Voigt fit and fill in only the rejected lines")
    ap.add_argument("--line_weights", default=None,
                    help="line_weights.h5 from scripts/select_cf_lines.py: per-line weights "
                         "multiplied into the classical weights (same dictionary required)")
    ap.add_argument("--sample", default=None, help="substring of the sample name/id")
    ap.add_argument("--r2_min", type=float, default=0.9)
    ap.add_argument("--isolation_min", type=float, default=None,
                    help="if set (and --line_dict given), keep only lines with isolation_score >= this")
    ap.add_argument("--T0", type=float, default=10000.0)
    ap.add_argument("--log10_Ne0", type=float, default=17.0)
    ap.add_argument("--log10_Nl0", type=float, default=16.0)
    ap.add_argument("--n_iter", type=int, default=3)
    ap.add_argument("--no_sa", action="store_true", help="disable the self-absorption correction")
    ap.add_argument("--prior_T", type=float, default=0.1)
    ap.add_argument("--prior_Ne", type=float, default=0.1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    table, _ = load_spectra_cache(args.spectra_cache)
    elements = _element_columns(table)
    n_total = len(table)
    cache_dir = Path(args.spectra_cache).resolve().parent
    idx = _resolve_indices(args.indices, table, args.sample, cache_dir, n_total)

    tables = build_cf_tables(elements, args.db, args.element_lod_config)
    lod = np.asarray(tables.lod, dtype=np.float64)

    with h5py.File(args.tokens, "r") as f:
        if int(f.attrs.get("n_spectra", f["tokens"].shape[0])) != n_total:
            raise SystemExit(f"token cache has {f['tokens'].shape[0]} spectra, spectra cache {n_total}")
        tok_ds, valid_ds = f["tokens"], f["fit_valid"]
        first = tok_ds[0]
        wl, Z, ion = first[:, _CH_WL], first[:, _CH_Z], first[:, _CH_ION]
        tokens = np.stack([tok_ds[int(i)] for i in idx]).astype(np.float64)
        fit_valid = np.stack([valid_ds[int(i)] for i in idx]).astype(np.float64)

    rescue_elems = None                      # which elements' blended lines may be rescued
    if args.line_weights:
        with h5py.File(args.line_weights, "r") as f:
            r = str(f.attrs.get("rescue", "all"))
            if r != "all":
                rescue_elems = np.isin(np.array([e.decode() for e in f["element"][:]]), r.split(","))
                print(f"line weights: rescuing blended lines of {r} only")
    if args.deconv:
        with h5py.File(args.deconv, "r") as f:
            d_area = f["area"][:].astype(np.float64)
            d_blend = f["blend_fraction"][:].astype(np.float64)
            d_rows = f["rows"][:].astype(np.int64)
        pos = {int(r): k for k, r in enumerate(d_rows)}
        missing = [int(i) for i in idx if int(i) not in pos]
        if missing:
            raise SystemExit(f"{args.deconv} has no rows for {len(missing)} requested spectra")
        take = np.stack([pos[int(i)] for i in idx])
        d_area, d_blend = d_area[take], d_blend[take]
        ok = (d_area > 0) & (np.nan_to_num(d_blend, nan=1.0) <= args.deconv_max_blend)
        if args.deconv_only_invalid or args.line_weights:      # never overwrite a good direct fit
            ok &= ~((fit_valid > 0.5) & (tokens[:, :, 11] >= args.r2_min))
        if rescue_elems is not None:
            ok &= rescue_elems[None, :]
        n_before = int((fit_valid > 0.5).sum())
        tokens[:, :, 9] = np.where(ok, d_area, tokens[:, :, 9])
        fit_valid = np.where(ok, 1.0, fit_valid)
        rescued_w = np.where(ok, 1.0 - np.nan_to_num(d_blend, nan=1.0), 0.0)   # weight of a rescued slot
        print(f"deconvolution: {int(ok.sum())} line-slots replaced "
              f"({ok.sum() / ok.size:.1%} of all); usable lines per spectrum "
              f"{n_before / len(idx):.0f} -> {(fit_valid > 0.5).sum() / len(idx):.0f}")

    isolation = forced = None
    if args.line_dict:
        with h5py.File(args.line_dict, "r") as f:
            if "isolation_score" in f:
                isolation = f["isolation_score"][:].astype(np.float64)
            if "forced" in f:
                forced = f["forced"][:].astype(np.float64)

    line_w = np.ones(wl.shape[0], dtype=np.float64)
    if args.line_weights:
        with h5py.File(args.line_weights, "r") as f:
            line_w = f["weight"][:].astype(np.float64)
            lw_wl = f["central_wavelength"][:]
        if line_w.shape[0] != wl.shape[0] or not np.allclose(lw_wl, wl, atol=1e-3):
            raise SystemExit(f"{args.line_weights} was made for a different line dictionary")
        print(f"line weights: {int((line_w > 0.5).sum())}/{line_w.size} lines above 0.5")

    line_mask = np.ones(wl.shape[0], dtype=bool)
    if args.line_list != "all":
        tsv = CF_OES_54_TSV if args.line_list == "cf_oes54" else CF_MINERAL_TSV
        line_mask = select_cf_oes_lines(wl, Z, ion, tsv_path=tsv)
        n_ref = len(load_cf_oes_lines(tsv))
        print(f"{args.line_list}: {int(line_mask.sum())}/{n_ref} curated lines found in the token cache")

    out_dir = Path(args.out) if args.out else Path("evaluation") / f"cf_classical_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    preds = np.zeros((len(idx), len(elements)), dtype=np.float64)
    t0 = time.time()
    for k, i in enumerate(idx):
        w = classical_weights(tokens[k], fit_valid[k], r2_min=args.r2_min,
                              isolation=isolation if args.isolation_min is not None else None,
                              forced=forced, isolation_min=args.isolation_min or 0.0)
        if args.deconv:                      # a rescued slot has no Voigt r²: weight it by its own share
            w = np.where(rescued_w[k] > 0, np.maximum(w, rescued_w[k]), w)
        w = w * line_mask * line_w
        res = saha_boltzmann_solve_np(
            tokens[k], fit_valid[k], w, tables, C0=None,
            T0=args.T0, log10_Ne0=args.log10_Ne0, log10_Nl0=args.log10_Nl0,
            n_iter=args.n_iter, prior_T=args.prior_T, prior_Ne=args.prior_Ne,
            sa_correction=not args.no_sa,
        )
        preds[k] = res.mass_fractions
        row = {"index": int(i), "unique_id": str(table["unique_id"].iloc[int(i)]),
               "sample": str(table["sample_type_name"].iloc[int(i)]),
               "T": float(res.T), "log10_Ne": float(res.log10_Ne),
               "n_lines": int(np.sum(res.used_mask))}
        for j, e in enumerate(elements):
            row[e] = float(res.mass_fractions[j])
            row[f"{e}_censored"] = bool(res.censored[j])
            row[f"{e}_n_lines"] = int(res.n_lines_used[j])
        rows.append(row)
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(idx)} spectra ({time.time() - t0:.0f} s)")
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_spectrum.csv", index=False)

    true = table[elements].to_numpy(dtype=np.float64)[idx]
    per_element: dict[str, dict] = {}
    for j, e in enumerate(elements):
        per_element[e] = {"spectrum": _metrics(preds[:, j], true[:, j], lod[j])}
    # per-sample medians (measured data: several shots per certified sample)
    samples = table["sample_type_id"].astype(str).to_numpy()[idx]
    uniq = np.unique(samples)
    if uniq.size >= 2:
        med_pred = np.stack([np.median(preds[samples == s], axis=0) for s in uniq])
        med_true = np.stack([np.median(true[samples == s], axis=0) for s in uniq])
        for j, e in enumerate(elements):
            per_element[e]["sample_median"] = _metrics(med_pred[:, j], med_true[:, j], lod[j])
    summary = {
        "tokens": str(args.tokens), "spectra_cache": str(args.spectra_cache),
        "line_list": args.line_list, "n_spectra": int(len(idx)), "n_samples": int(uniq.size),
        "T_median": float(np.median(df["T"])), "log10_Ne_median": float(np.median(df["log10_Ne"])),
        "n_lines_median": float(np.median(df["n_lines"])),
        "per_element": per_element,
    }
    with open(out_dir / "per_element.yaml", "w") as fh:
        yaml.safe_dump(summary, fh, sort_keys=False)

    print(f"\n{len(idx)} spectra, {uniq.size} samples; median T {summary['T_median']:.0f} K, "
          f"median log10 Ne {summary['log10_Ne_median']:.2f}, median lines used {summary['n_lines_median']:.0f}")
    print(f"{'El':>3s} {'true_med':>9s} {'pred_med':>9s} {'mae':>9s} {'log_rmse':>8s} {'within2x':>8s} {'r2':>7s}")
    for j, e in enumerate(elements):
        m = per_element[e]["spectrum"]
        if m["n_above_lod"] == 0:
            continue
        print(f"{e:>3s} {np.median(true[:, j]):9.4f} {np.median(preds[:, j]):9.4f} {m['mae']:9.4f} "
              f"{m.get('log_rmse', float('nan')):8.3f} {m.get('within_2x', float('nan')):8.3f} "
              f"{m.get('r2', float('nan')):7.3f}")
    print(f"\nwritten: {out_dir}")


if __name__ == "__main__":
    main()
