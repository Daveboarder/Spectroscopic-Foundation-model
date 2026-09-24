"""
Add missing rare-earth lines to the line DB (LIBS_data.db) from NIST ASD and laboratory data.

Sources (fetched once into ``external_data/cache/line_db_sources/``):

* NIST ASD lines (https://physics.nist.gov/cgi-bin/ASD/lines1.pl), stages I and II of Y and the
  lanthanides, 188-860 nm; only lines with Aki, both level energies and both g values (levels
  with an unknown offset, "+x", are rejected). Same
  wavelength convention as the DB (NIST default: vacuum below 200 nm, air above).
* Wisconsin laboratory transition probabilities (radiative lifetimes + FTS branching fractions,
  typically +/-5-10 %) from VizieR, for lines NIST lists without Aki:
    Gd II  Den Hartog et al. 2006, ApJS 167, 292      (J/ApJS/167/292)
    Sm II  Lawler et al. 2006, ApJS 162, 227           (J/ApJS/162/227/table2)
    Ce II  Lawler et al. 2009, ApJS 182, 51            (J/ApJS/182/51/table2)
    La II  Lawler et al. 2001, ApJ 556, 452   (via J/ApJS/182/51/table5)
    Eu II  Lawler et al. 2001, ApJ 563, 1075  (via J/ApJS/182/51/table6)
    Dy I/II Wickliffe et al. 2000, JQSRT 66, 363 (via J/ApJS/182/51/table8)
    Ho II  Lawler et al. 2004, ApJ 604, 850   (via J/ApJS/182/51/table9)

Only lines missing from the DB are inserted; existing rows are never changed (NIST values win).
A line is "already present" when the DB (or an earlier source in this run) has the same element
and stage with |d lambda| <= 0.01 nm and the same upper level (|d Ek| <= 5 meV). New rows are
recorded in the table ``QuantParam_source`` (qp_id, source, accuracy); QuantParam is unchanged in
layout. Units written: wavelength nm, Ei/Ek eV, Ak s^-1, gi/gk = 2J+1.

Usage:
    uv run python scripts/augment_line_db.py --dry-run
    uv run python scripts/augment_line_db.py --apply
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sqlite3
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_DEFAULT = ROOT / "external_data" / "Source" / "LIBS_data.db"
CACHE = ROOT / "external_data" / "cache" / "line_db_sources"

CM1_TO_EV = 1.0 / 8065.543937  # same factor as the existing NIST-derived rows (|dE| ~ 1e-9 eV)
WL_TOL_NM = 0.01
EK_TOL_EV = 5e-3

NIST_ELEMENTS = ("Y", "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm",
                 "Yb", "Lu")
NIST_RANGE_NM = (188.0, 860.0)

# (source label, VizieR table, element, stage or None = from J parity, column names)
LAB_TABLES = (
    ("DenHartog2006_GdII", "J/ApJS/167/292", "Gd", "II",
     dict(wl="lambda", eu="E1", ju="J1", el="E0", jl="J0", a="A(1-0)", acc="e_TranP")),
    ("Lawler2006_SmII", "J/ApJS/162/227/table2", "Sm", "II",
     dict(wl="Lambda", eu="Eu", ju="Ju", el="El", jl="Jl", a="TransP", acc="e_TransP")),
    ("Lawler2009_CeII", "J/ApJS/182/51/table2", "Ce", "II",
     dict(wl="lamAir", eu="E1", ju="J1", el="E0", jl="J0", a="Aij", acc="e_Aij")),
    ("Lawler2001_LaII", "J/ApJS/182/51/table5", "La", "II",
     dict(wl="lamAir", eu="E1", ju="J1", el="E0", jl="J0", a="Aij", acc="e_Aij")),
    ("Lawler2001_EuII", "J/ApJS/182/51/table6", "Eu", "II",
     dict(wl="lamAir", eu="E1", ju="J1", el="E0", jl="J0", a="Aij", acc="e_Aij")),
    # Dy: integral J = Dy I, half-integral J = Dy II (catalogue ReadMe).
    ("Wickliffe2000_Dy", "J/ApJS/182/51/table8", "Dy", None,
     dict(wl="lamAir", eu="E1", ju="J1", el="E0", jl="J0", a="AijW", acc="e_AijW")),
    ("Lawler2004_HoII", "J/ApJS/182/51/table9", "Ho", "II",
     dict(wl="lamAir", eu="E1", ju="J1", el="E0", jl="J0", a="Aij", acc="e_Aij")),
)
LAB_RANGE_NM = (200.0, 1000.0)
LAB_A_UNIT = 1e6        # VizieR columns are in 10^6 s^-1
LAB_WL_UNIT = 0.1       # Angstrom -> nm (air)


def _fetch(url: str, path: Path) -> str:
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        # NIST rejects urllib's default User-Agent with HTTP 403.
        req = urllib.request.Request(url, headers={"User-Agent": "augment_line_db/1.0 (research)"})
        with urllib.request.urlopen(req, timeout=180) as r:
            path.write_bytes(r.read())
        time.sleep(1.0)  # be polite to NIST / CDS
    return path.read_text(encoding="utf-8")


def _num(s: str | None) -> float | None:
    """NIST cell -> float; keeps exponents, drops markers ([] () ? * =) and letter flags."""
    s = re.sub(r"[\[\]\(\)\?\*=\s]", "", (s or "").strip().strip('"'))
    s = re.sub(r"(?<=\d)[a-df-zA-DF-Z]+$", "", s)
    try:
        return float(s)
    except ValueError:
        return None


def nist_lines(element: str, stage: str) -> list[dict]:
    q = {
        "spectra": f"{element} {stage}", "limits_type": 0, "low_w": NIST_RANGE_NM[0],
        "upp_w": NIST_RANGE_NM[1], "unit": 1, "submit": "Retrieve Data", "de": 0, "format": 3,
        "line_out": 0, "en_unit": 0, "output": 0, "bibrefs": 1, "page_size": 15,
        "show_obs_wl": 1, "show_calc_wl": 1, "unc_out": 1, "order_out": 0, "show_av": 2,
        "tsb_value": 0, "A_out": 0, "intens_out": "on", "allowed_out": 1, "forbid_out": 1,
        "conf_out": "on", "term_out": "on", "enrg_out": "on", "J_out": "on", "g_out": "on",
    }
    url = "https://physics.nist.gov/cgi-bin/ASD/lines1.pl?" + urllib.parse.urlencode(q)
    txt = _fetch(url, CACHE / f"nist_{element}_{stage}.tsv")
    if not txt.startswith("obs_wl"):
        raise RuntimeError(f"NIST query for {element} {stage} did not return a line table")
    out = []
    for r in csv.DictReader(io.StringIO(txt), delimiter="\t"):
        r = {k: (v or "").strip().strip('"') for k, v in r.items() if k}
        wl = _num(r.get("obs_wl_air(nm)")) or _num(r.get("ritz_wl_air(nm)"))
        A, Ei, Ek = _num(r.get("Aki(s^-1)")), _num(r.get("Ei(cm-1)")), _num(r.get("Ek(cm-1)"))
        gi, gk = _num(r.get("g_i")), _num(r.get("g_k"))
        if None in (wl, A, Ei, Ek, gi, gk) or A <= 0:
            continue
        out.append(dict(el=element, st=stage, wl=wl, Ei=Ei * CM1_TO_EV, Ek=Ek * CM1_TO_EV,
                        A=A, gi=gi, gk=gk, source="NIST_ASD", acc=r.get("Acc", "")))
    return out


def lab_lines(label: str, table: str, element: str, stage: str | None, c: dict) -> list[dict]:
    url = ("https://vizier.cds.unistra.fr/viz-bin/asu-tsv?"
           + urllib.parse.urlencode({"-source": table, "-out.max": "unlimited", "-out.all": ""}))
    txt = _fetch(url, CACHE / f"vizier_{table.replace('/', '_')}.tsv")
    lines = [ln.rstrip("\n").split("\t") for ln in txt.splitlines()
             if ln.strip() and not ln.startswith("#")]
    hdr = [h.strip() for h in lines[0]]
    out = []
    for row in lines[3:]:  # header, units, dashes
        r = dict(zip(hdr, (x.strip() for x in row)))
        try:
            wl = float(r[c["wl"]]) * LAB_WL_UNIT
            A = float(r[c["a"]]) * LAB_A_UNIT
            Eu, Ju = float(r[c["eu"]]) * CM1_TO_EV, float(r[c["ju"]])
            El, Jl = float(r[c["el"]]) * CM1_TO_EV, float(r[c["jl"]])
        except (KeyError, ValueError):
            continue
        # Lab tables give air wavelengths (DB: vacuum below 200 nm); DB range ends at ~1000 nm.
        if not LAB_RANGE_NM[0] <= wl <= LAB_RANGE_NM[1] or A <= 0:
            continue
        st = stage or ("I" if Ju.is_integer() else "II")
        e_A = _num(r.get(c["acc"]))
        acc = f"+/-{100 * e_A / float(r[c['a']]):.0f}%" if e_A else ""
        out.append(dict(el=element, st=st, wl=wl, Ei=El, Ek=Eu, A=A, gi=2 * Jl + 1,
                        gk=2 * Ju + 1, source=label, acc=acc))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", default=str(DB_DEFAULT))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    candidates: list[dict] = []
    for el in NIST_ELEMENTS:  # NIST first: it wins over lab data for the same transition
        for st in ("I", "II"):
            candidates += nist_lines(el, st)
    for spec in LAB_TABLES:
        candidates += lab_lines(*spec)

    con = sqlite3.connect(args.db)
    have: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for el, st, wl, ek in con.execute("SELECT Elem_name, ion_state, Wavelength, Ek FROM QuantParam"):
        have.setdefault((el, st), []).append((wl, ek))

    new: list[dict] = []
    for c in candidates:
        known = have.setdefault((c["el"], c["st"]), [])
        if any(abs(w - c["wl"]) <= WL_TOL_NM and abs(e - c["Ek"]) <= EK_TOL_EV for w, e in known):
            continue
        known.append((c["wl"], c["Ek"]))
        new.append(c)

    summary: dict[tuple[str, str, str], int] = {}
    for c in new:
        k = (c["source"], c["el"], c["st"])
        summary[k] = summary.get(k, 0) + 1
    for (src, el, st), n in sorted(summary.items()):
        print(f"  {src:22s} {el:2s} {st:2s} +{n}")
    print(f"{len(new)} new lines of {len(candidates)} candidates")

    if args.dry_run:
        return
    with con:
        con.execute("CREATE TABLE IF NOT EXISTS QuantParam_source ("
                    "qp_id INTEGER PRIMARY KEY REFERENCES QuantParam(id), "
                    "source TEXT NOT NULL, accuracy TEXT)")
        for c in new:
            cur = con.execute(
                "INSERT INTO QuantParam (Elem_name, ion_state, Wavelength, Ei, Ek, Ak, gi, gk) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (c["el"], c["st"], round(c["wl"], 4), c["Ei"], c["Ek"], c["A"], c["gi"], c["gk"]),
            )
            con.execute("INSERT INTO QuantParam_source VALUES (?, ?, ?)",
                        (cur.lastrowid, c["source"], c["acc"]))
    print(f"inserted {len(new)} rows into {args.db}")


if __name__ == "__main__":
    main()
