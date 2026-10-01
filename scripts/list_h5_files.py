"""
List every HDF5 file (*.h5, *.hdf5) in a directory tree, by default
/mnt/data/projects/Running_projects, count the spectra in each, and write the
list plus a summary with the totals.

Columns of the CSV:
    path, project (first directory below the root), size_mb, modified,
    layout, n_spectra, n_measurements, n_pixels, oem, error
(with --wavelengths also wl_min_nm, wl_max_nm).

How spectra are counted (only dataset shapes are read, never the data):
    * layout "lightigo": LIGHTIGO / FireFly files, measurements/<key>/libs/data
      of shape (n_spectra, n_pixels), summed over all measurements;
    * layout "generic": any other file - every 2-D dataset whose name contains
      "spectr" (e.g. Avantes .../ShotSpectra) contributes its first dimension;
    * layout "unknown": no spectrum-like dataset found -> n_spectra empty,
      not included in the total; the summary lists how many such files exist;
    * files that cannot be opened get the error text in `error`.

Outputs (default names in Outputs/, gitignored):
    h5_files_<ts>.csv          one row per file
    h5_files_<ts>_summary.txt  totals (files, spectra, size) overall, per layout
                               and per project, plus unreadable directories

The walk does not follow symbolic links, skips unreadable directories and
ignores virtual environments, .git and __pycache__ folders.  Walking the
network share takes ~15 min; --from_csv reuses the file list of an earlier run
(paths, sizes and dates are taken from it) and only opens the files.

Usage:
    uv run python scripts/list_h5_files.py
    uv run python scripts/list_h5_files.py --from_csv Outputs/h5_files_2026-09-25_11-06-31.csv
    uv run python scripts/list_h5_files.py --root /mnt/data/projects/Running_projects --workers 16 --wavelengths
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

DEFAULT_ROOT = "/mnt/data/projects/Running_projects"
EXTENSIONS = (".h5", ".hdf5")
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", ".ipynb_checkpoints"}
BASE_FIELDS = ["path", "project", "size_mb", "modified"]
INFO_FIELDS = ["layout", "n_spectra", "n_measurements", "n_pixels", "oem", "error"]
WL_FIELDS = ["wl_min_nm", "wl_max_nm"]


# ─────────────────────────────────────────────────────────────────────────────
# File discovery
# ─────────────────────────────────────────────────────────────────────────────
def find_h5(root: Path, errors: list[str]):
    """Yield os.DirEntry objects of HDF5 files below ``root`` (iterative walk)."""
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            if e.name not in SKIP_DIRS:
                                stack.append(Path(e.path))
                        elif e.is_file(follow_symlinks=False) and e.name.lower().endswith(
                            EXTENSIONS
                        ):
                            yield e
                    except OSError as exc:
                        errors.append(f"{e.path}: {exc}")
        except OSError as exc:
            errors.append(f"{d}: {exc}")


def walk_rows(root: Path, errors: list[str], t0: float) -> list[dict]:
    rows = []
    for e in find_h5(root, errors):
        try:
            st = e.stat(follow_symlinks=False)
        except OSError as exc:
            errors.append(f"{e.path}: {exc}")
            continue
        rel = Path(e.path).relative_to(root)
        rows.append(
            dict(
                path=e.path,
                project=rel.parts[0] if len(rel.parts) > 1 else "",
                size_mb=round(st.st_size / 2**20, 3),
                modified=datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            )
        )
        if len(rows) % 500 == 0:
            print(f"  walk: {len(rows)} files ({time.perf_counter() - t0:.0f} s)", flush=True)
    return rows


def rows_from_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return [{k: r[k] for k in BASE_FIELDS} for r in csv.DictReader(f)]


# ─────────────────────────────────────────────────────────────────────────────
# Spectrum counting
# ─────────────────────────────────────────────────────────────────────────────
def inspect(args: tuple[str, bool]) -> dict:
    """Layout, spectrum count and header of one HDF5 file (shapes only)."""
    path, want_wl = args
    import h5py

    info = dict(layout="", n_spectra="", n_measurements="", n_pixels="", oem="", error="")
    if want_wl:
        info.update(wl_min_nm="", wl_max_nm="")
    try:
        with h5py.File(path, "r") as f:
            if "OEM" in f:
                v = f["OEM"][()]
                v = v[0] if hasattr(v, "__len__") and not isinstance(v, (bytes, str)) else v
                info["oem"] = v.decode(errors="replace") if isinstance(v, bytes) else str(v)

            # LIGHTIGO / FireFly: measurements/<key>/libs/data
            n_spec, n_px, n_meas, lo, hi = 0, None, 0, None, None
            if "measurements" in f and isinstance(f["measurements"], h5py.Group):
                for key in f["measurements"]:
                    g = f["measurements"][key]
                    libs = g.get("libs") if isinstance(g, h5py.Group) else None
                    if not isinstance(libs, h5py.Group) or "data" not in libs:
                        continue
                    shape = libs["data"].shape
                    n_meas += 1
                    n_spec += shape[0] if len(shape) > 1 else 1
                    n_px = shape[-1]
                    if want_wl and "calibration" in libs:
                        wl = libs["calibration"][...]
                        lo = float(wl.min()) if lo is None else min(lo, float(wl.min()))
                        hi = float(wl.max()) if hi is None else max(hi, float(wl.max()))
            if n_meas:
                info.update(
                    layout="lightigo", n_spectra=n_spec, n_measurements=n_meas, n_pixels=n_px
                )
                if want_wl and lo is not None:
                    info.update(wl_min_nm=round(lo, 2), wl_max_nm=round(hi, 2))
                return info

            # any other layout: 2-D datasets named like spectra
            found = {"n": 0, "sets": 0, "px": None}

            def visit(name, obj):
                if (
                    isinstance(obj, h5py.Dataset)
                    and len(obj.shape) == 2
                    and "spectr" in name.rsplit("/", 1)[-1].lower()
                ):
                    found["n"] += int(obj.shape[0])
                    found["sets"] += 1
                    found["px"] = int(obj.shape[1])

            f.visititems(visit)
            if found["sets"]:
                info.update(
                    layout="generic",
                    n_spectra=found["n"],
                    n_measurements=found["sets"],
                    n_pixels=found["px"],
                )
            else:
                info["layout"] = "unknown"
    except Exception as exc:  # corrupted / locked / non-HDF5 file
        info["layout"] = "error"
        info["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return info


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
def _fmt_size(mb: float) -> str:
    return f"{mb / 2**20:.2f} TB" if mb >= 2**20 else f"{mb / 1024:.1f} GB"


def summary_text(rows: list[dict], errors: list[str], root: str, elapsed: float) -> str:
    tot_mb = sum(float(r["size_mb"]) for r in rows)
    counted = [r for r in rows if r.get("n_spectra") not in ("", None)]
    tot_spec = sum(int(r["n_spectra"]) for r in counted)
    by_layout: dict[str, list] = defaultdict(lambda: [0, 0, 0.0])
    by_proj: dict[str, list] = defaultdict(lambda: [0, 0, 0.0])
    for r in rows:
        n = int(r["n_spectra"]) if r.get("n_spectra") not in ("", None) else 0
        for key, table in ((r.get("layout", ""), by_layout), (r["project"] or "(root)", by_proj)):
            table[key][0] += 1
            table[key][1] += n
            table[key][2] += float(r["size_mb"])

    lines = [
        f"HDF5 inventory of {root}  ({datetime.now():%Y-%m-%d %H:%M}, {elapsed:.0f} s)",
        "",
        f"TOTAL files:    {len(rows):,}",
        f"TOTAL spectra:  {tot_spec:,}  (in {len(counted):,} files with a recognised layout)",
        f"TOTAL size:     {_fmt_size(tot_mb)}  ({tot_mb:,.0f} MB)",
        "",
        "by layout:  files | spectra | size",
    ]
    for k, (nf, ns, mb) in sorted(by_layout.items(), key=lambda kv: -kv[1][0]):
        lines.append(f"  {k or '-':10s} {nf:7,} | {ns:13,} | {_fmt_size(mb)}")
    lines += ["", "by project:  files | spectra | size"]
    for k, (nf, ns, mb) in sorted(by_proj.items(), key=lambda kv: -kv[1][1]):
        lines.append(f"  {k:45s} {nf:6,} | {ns:13,} | {_fmt_size(mb)}")
    if errors:
        lines += ["", f"{len(errors)} paths could not be read (not included above):"]
        lines += [f"  {m}" for m in errors]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default=DEFAULT_ROOT, help="top directory to search")
    ap.add_argument("--out", default=None, help="CSV path (default Outputs/h5_files_<ts>.csv)")
    ap.add_argument(
        "--from_csv", default=None, help="reuse the file list of an earlier run instead of walking"
    )
    ap.add_argument("--workers", type=int, default=12, help="parallel processes opening files")
    ap.add_argument(
        "--wavelengths", action="store_true", help="also read the LIGHTIGO wavelength range"
    )
    args = ap.parse_args()

    root = Path(args.root)
    repo = Path(__file__).resolve().parent.parent
    out = (
        Path(args.out)
        if args.out
        else repo / "Outputs" / f"h5_files_{datetime.now():%Y-%m-%d_%H-%M-%S}.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    errors: list[str] = []
    if args.from_csv:
        rows = rows_from_csv(Path(args.from_csv))
        print(f"{len(rows)} files taken from {args.from_csv}")
    else:
        if not root.is_dir():
            sys.exit(f"root directory not found: {root}")
        rows = walk_rows(root, errors, t0)
        print(f"walk: {len(rows)} HDF5 files in {time.perf_counter() - t0:.0f} s")

    t1 = time.perf_counter()
    jobs = [(r["path"], args.wavelengths) for r in rows]
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for i, (r, info) in enumerate(zip(rows, pool.map(inspect, jobs, chunksize=8)), 1):
            r.update(info)
            if i % 500 == 0:
                print(
                    f"  inspect: {i}/{len(rows)} files ({time.perf_counter() - t1:.0f} s)",
                    flush=True,
                )
    print(f"inspect: {len(rows)} files in {time.perf_counter() - t1:.0f} s")

    fields = BASE_FIELDS + INFO_FIELDS + (WL_FIELDS if args.wavelengths else [])
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    text = summary_text(rows, errors, str(root), time.perf_counter() - t0)
    summary_path = out.with_name(out.stem + "_summary.txt")
    summary_path.write_text(text, encoding="utf-8")
    print("\n" + "\n".join(text.splitlines()[:12]))
    print(f"\n[results] {out}\n          {summary_path}")


if __name__ == "__main__":
    main()
