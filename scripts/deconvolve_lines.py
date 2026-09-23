"""
Deconvolve the blended lines of a spectra cache at a known plasma state.

The isolation filter throws away lines that overlap a neighbour.  They are not
unmeasurable: the central wavelengths in the window are known (the line
database), the plasma state is known (the Saha-Boltzmann solve, or a fixed
value), and the profile is known (Doppler at T, Stark as a Lorentzian, the
instrument function).  ``cf.deconv`` fits the whole window as a sum of those
profiles with one free scale per element, which is linear and well posed, and
returns each line's own area separated from the blend.

Writes ``deconv_<hash>.h5``:

    area            [n_spectra, n_lines]  deconvolved integrated area
    blend_fraction  [n_spectra, n_lines]  1 - the line's own share of its peak
    valid           [n_spectra, n_lines]  area > 0 and blend below --max_blend
    window_r2       [n_spectra, n_lines]  fit quality of the window it came from

Usage:
    uv run python scripts/deconvolve_lines.py \\
        --spectra_cache external_data/cache/measured_cache_<h>.h5 \\
        --line_dict external_data/cache/line_dict_<h>.h5 \\
        --plasma_csv runs/<cf run>/evaluation/cf_measured_<ts>/per_spectrum.csv \\
        --workers 22 --indices all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cf.deconv import DeconvConfig, _LineSetCache, deconvolve_window  # noqa: E402
from data.libs_pipeline import load_spectra_cache, load_wavelength  # noqa: E402

_W: dict = {}


def _init(wl, spectra, centres, elems, elements, cfg, db_path, t_step, ne_step,
          shift=None, sess_idx=None):
    _W.update(wl=wl, spectra=spectra, centres=centres, elems=elems, elements=elements,
              cfg=cfg, cache=_LineSetCache(db_path), t_step=t_step, ne_step=ne_step,
              shift=shift, sess_idx=sess_idx)


def _quantise(T: float, log_ne: float, t_step: float, ne_step: float) -> tuple[float, float]:
    """Snap the plasma state to a grid so the line-set cache stays small; the
    Boltzmann weights vary far more slowly than these steps."""
    return (round(float(T) / t_step) * t_step, round(float(log_ne) / ne_step) * ne_step)


def _one(task):
    """Deconvolve every target line of one spectrum, window by window."""
    row, T, log_ne = task
    wl, spectra, centres, elems = _W["wl"], _W["spectra"], _W["centres"], _W["elems"]
    cfg, cache, elements = _W["cfg"], _W["cache"], _W["elements"]
    Tq, nq = _quantise(T, log_ne, _W["t_step"], _W["ne_step"])
    Ne = 10.0 ** nq
    s = spectra[row].astype(np.float64)
    if _W.get("shift") is not None:                 # per-session axis correction
        wl = wl - _W["shift"][_W["sess_idx"][row]]

    n = centres.size
    area = np.zeros(n, dtype=np.float32)
    blend = np.full(n, np.nan, dtype=np.float32)
    win_r2 = np.full(n, np.nan, dtype=np.float32)

    order = np.argsort(centres)
    i = 0
    while i < order.size:
        group = [order[i]]
        while (i + 1 < order.size
               and centres[order[i + 1]] - centres[group[0]] <= cfg.window_nm):
            i += 1
            group.append(order[i])
        i += 1
        centre = float(np.mean(centres[group]))
        targets = [(elems[j], float(centres[j])) for j in group]
        res = deconvolve_window(wl, s, centre, elements, Tq, Ne, cache, cfg, targets=targets)
        r2 = res.r2
        for k, j in enumerate(group):
            a = res.area[k]
            area[j] = 0.0 if not np.isfinite(a) else max(float(a), 0.0)
            if res.blend_fraction is not None and np.isfinite(res.blend_fraction[k]):
                blend[j] = float(res.blend_fraction[k])
            win_r2[j] = r2
    return row, area, blend, win_r2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spectra_cache", required=True)
    ap.add_argument("--line_dict", required=True)
    ap.add_argument("--plasma_csv", default=None,
                    help="per_spectrum.csv with cf_T / cf_log10_Ne (else --T / --log10_Ne)")
    ap.add_argument("--T", type=float, default=11000.0)
    ap.add_argument("--log10_Ne", type=float, default=17.5)
    ap.add_argument("--wavelength_json", default="external_data/Data/VASKUT K8.json")
    ap.add_argument("--db", default="external_data/Source/LIBS_data.db")
    ap.add_argument("--window_nm", type=float, default=0.30)
    ap.add_argument("--margin_nm", type=float, default=0.60)
    ap.add_argument("--instrument_fwhm_nm", type=float, default=0.047)
    ap.add_argument("--gamma_nm", type=float, default=0.010)
    ap.add_argument("--max_blend", type=float, default=0.8,
                    help="a line whose peak is more than this fraction other lines is not trusted")
    ap.add_argument("--mode", choices=("per_element", "per_line", "hybrid"), default="per_element",
                    help="hybrid: target lines free, other lines of the element tied by theory")
    ap.add_argument("--t_step", type=float, default=250.0)
    ap.add_argument("--ne_step", type=float, default=0.1)
    ap.add_argument("--indices", default="all", help="all | <n first>")
    ap.add_argument("--axis_from_tokens", default=None,
                    help="line_tokens_*.h5 of this cache: correct the axis per session from the "
                         "Voigt-fit centre offsets (cf.axis.session_axis_shift)")
    ap.add_argument("--axis_break_nm", type=float, default=250.0)
    ap.add_argument("--axis_deg", type=int, default=3)
    ap.add_argument("--workers", type=int, default=22)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    table, spectra = load_spectra_cache(args.spectra_cache)
    wl = load_wavelength(args.wavelength_json)
    with h5py.File(args.line_dict, "r") as f:
        centres = f["central_wavelength"][:].astype(np.float64)
        # `element_id` is a vocabulary index, NOT the atomic number - the symbols
        # live in vocab/element (token channel 7 is the real Z)
        raw = f["vocab"]["element"][:]
    elems = np.array([v.decode() if isinstance(v, (bytes, bytearray)) else str(v) for v in raw])
    keep = elems != "?"
    if not np.all(keep):
        print(f"WARN: {int((~keep).sum())} lines have an unmapped element id")
    elements = sorted(set(elems[keep]))
    print(f"{len(centres)} target lines, {len(elements)} elements, "
          f"{spectra.shape[0]} spectra on {wl.size} channels")

    n_total = spectra.shape[0]
    rows = np.arange(n_total) if args.indices == "all" else np.arange(min(int(args.indices), n_total))

    if args.plasma_csv:
        df = pd.read_csv(args.plasma_csv)
        T_all = df["cf_T"].to_numpy(dtype=np.float64)
        ne_all = df["cf_log10_Ne"].to_numpy(dtype=np.float64)
        if T_all.size != n_total:
            raise SystemExit(f"{args.plasma_csv} has {T_all.size} rows, cache has {n_total}")
        print(f"plasma state from {args.plasma_csv}: median T {np.median(T_all):.0f} K, "
              f"median log10 Ne {np.median(ne_all):.2f}")
    else:
        T_all = np.full(n_total, args.T)
        ne_all = np.full(n_total, args.log10_Ne)
        print(f"fixed plasma state: T {args.T:.0f} K, log10 Ne {args.log10_Ne:.2f}")

    shift = sess_idx = None
    if args.axis_from_tokens:
        from cf.axis import session_axis_shift, sessions_from_unique_id
        with h5py.File(args.axis_from_tokens, "r") as f:
            tok = f["tokens"][:]; fvalid = f["fit_valid"][:]
        if tok.shape[0] != n_total:
            raise SystemExit(f"{args.axis_from_tokens} has {tok.shape[0]} spectra, cache has {n_total}")
        with h5py.File(args.line_dict, "r") as f:
            iso_l = f["isolation_score"][:] if "isolation_score" in f else None
        sessions = sessions_from_unique_id(table["unique_id"].to_numpy())
        shift, sess_idx, st = session_axis_shift(tok, fvalid, sessions, wl, isolation=iso_l,
                                                 break_nm=args.axis_break_nm, deg=args.axis_deg)
        del tok, fvalid
        print(f"axis correction: {st['n_sessions']} sessions, {st['lines_per_session']:.0f} lines/session, "
              f"|offset| median {st['offset_before_median_nm']:.4f} -> {st['resid_median_nm']:.4f} nm "
              f"(p90 {st['resid_p90_nm']:.4f})")

    cfg = DeconvConfig(window_nm=args.window_nm, margin_nm=args.margin_nm,
                       instrument_fwhm_nm=args.instrument_fwhm_nm,
                       gamma_nm=args.gamma_nm, mode=args.mode)

    key = hashlib.md5(json.dumps({
        "spectra": Path(args.spectra_cache).name, "dict": Path(args.line_dict).name,
        "cfg": cfg.__dict__, "t_step": args.t_step, "ne_step": args.ne_step,
        "plasma": Path(args.plasma_csv).name if args.plasma_csv else [args.T, args.log10_Ne],
        "n": int(rows.size),
        "axis": [Path(args.axis_from_tokens).name, args.axis_break_nm, args.axis_deg] if args.axis_from_tokens else None,
    }, sort_keys=True, default=str).encode()).hexdigest()[:12]
    out_path = Path(args.out or f"external_data/cache/deconv_{key}.h5")

    n, L = rows.size, centres.size
    area = np.zeros((n, L), dtype=np.float32)
    blend = np.full((n, L), np.nan, dtype=np.float32)
    win_r2 = np.full((n, L), np.nan, dtype=np.float32)
    tasks = [(int(r), float(T_all[r]), float(ne_all[r])) for r in rows]
    pos = {int(r): k for k, r in enumerate(rows)}

    t0 = time.time()
    if args.workers > 1 and n > 1:
        ctx = mp.get_context("fork")            # share `spectra` copy-on-write
        with ctx.Pool(args.workers, initializer=_init,
                      initargs=(wl, spectra, centres, elems, elements, cfg, args.db,
                                args.t_step, args.ne_step, shift, sess_idx)) as pool:
            for done, (row, a, b, r2) in enumerate(
                    pool.imap_unordered(_one, tasks, chunksize=4), start=1):
                k = pos[row]
                area[k], blend[k], win_r2[k] = a, b, r2
                if done % max(1, n // 20) == 0:
                    print(f"  {done}/{n} spectra ({time.time() - t0:.0f} s)", flush=True)
    else:
        _init(wl, spectra, centres, elems, elements, cfg, args.db, args.t_step, args.ne_step, shift, sess_idx)
        for done, task in enumerate(tasks, start=1):
            row, a, b, r2 = _one(task)
            k = pos[row]
            area[k], blend[k], win_r2[k] = a, b, r2
            if done % max(1, n // 20) == 0:
                print(f"  {done}/{n} spectra ({time.time() - t0:.0f} s)", flush=True)

    valid = (area > 0) & (np.nan_to_num(blend, nan=1.0) <= args.max_blend)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("area", data=area, compression="gzip", compression_opts=4)
        f.create_dataset("blend_fraction", data=blend, compression="gzip", compression_opts=4)
        f.create_dataset("valid", data=valid.astype(np.uint8), compression="gzip", compression_opts=4)
        f.create_dataset("window_r2", data=win_r2, compression="gzip", compression_opts=4)
        f.create_dataset("rows", data=rows.astype(np.int64))
        f.attrs["spectra_cache"] = str(args.spectra_cache)
        f.attrs["line_dict"] = str(args.line_dict)
        f.attrs["config_json"] = json.dumps(cfg.__dict__, default=str)
        f.attrs["max_blend"] = float(args.max_blend)
        f.attrs["axis_from_tokens"] = str(args.axis_from_tokens or "")
        f.attrs["mode"] = str(args.mode)

    print(f"\n{n} spectra in {time.time() - t0:.0f} s")
    print(f"  usable lines per spectrum: median {np.median(valid.sum(axis=1)):.0f} "
          f"of {L} (blend <= {args.max_blend})")
    print(f"  median blend fraction {np.nanmedian(blend):.2f}, "
          f"median window r2 {np.nanmedian(win_r2):.3f}")
    print(f"  written: {out_path}")


if __name__ == "__main__":
    main()
