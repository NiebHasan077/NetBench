#!/usr/bin/env python3
"""Compare regenerated analysis outputs against the committed ones by content.

Two of the formats the analysis emits stamp the current time into the file, so
a byte comparison reports a difference on every run even when nothing changed:

  * PDF   -- ``/CreationDate``, ``/ModDate``, ``/ID`` written by matplotlib;
  * XLSX  -- ``dcterms:created`` / ``dcterms:modified`` in ``docProps/core.xml``
             written by openpyxl.

For PDF the timestamp also has to be removed *before* the file's byte offsets
mean anything. matplotlib writes the local UTC offset into ``/CreationDate``,
so the field is 23 bytes at ``-05'00'`` and 17 at ``Z``; everything after it
shifts, and the cross-reference table and ``startxref`` disagree by exactly
that difference on two machines in different timezones. Those offsets are
derived bookkeeping -- reconstructible from the objects around them -- so they
are dropped rather than compared. The object and stream bytes still are.

Everything else (CSV, Markdown, LaTeX) is compared byte for byte, with no
normalisation, because those must be exactly reproducible.

Exit status is 0 only if every file matches, so this is usable as a gate:

    python tools/compare_outputs.py analysis/outputs/figures paper_eacl/figures
    make verify
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import zipfile
from pathlib import Path

PDF_STAMP = re.compile(rb"/(CreationDate|ModDate|ID)[^\n]*")
PDF_XREF = re.compile(rb"\nxref\n.*?\ntrailer\n", re.S)
PDF_STARTXREF = re.compile(rb"\nstartxref\n\d+\n")
XLSX_STAMP = re.compile(rb"<dcterms:(created|modified)[^>]*>[^<]*</dcterms:(created|modified)>")


def digest(path: Path) -> str:
    """Content hash, ignoring the embedded timestamps of PDF and XLSX."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        data = PDF_STAMP.sub(b"", path.read_bytes())
        data = PDF_XREF.sub(b"\ntrailer\n", data)
        return hashlib.sha256(PDF_STARTXREF.sub(b"\n", data)).hexdigest()
    if suffix == ".xlsx":
        h = hashlib.sha256()
        with zipfile.ZipFile(path) as z:
            for name in sorted(z.namelist()):
                data = z.read(name)
                if name == "docProps/core.xml":
                    data = XLSX_STAMP.sub(b"", data)
                h.update(name.encode()); h.update(data)
        return h.hexdigest()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("new", help="directory of freshly generated files")
    ap.add_argument("committed", help="directory of the files of record")
    ap.add_argument("--glob", default="*", help="restrict to matching names")
    ap.add_argument("--quiet", action="store_true", help="print only the summary")
    a = ap.parse_args(argv)

    new, old = Path(a.new), Path(a.committed)
    files = sorted(p for p in new.glob(a.glob) if p.is_file())
    if not files:
        print(f"nothing to compare in {new}/{a.glob}", file=sys.stderr)
        return 2

    same, missing, differ = [], [], []
    for f in files:
        counterpart = old / f.name
        if not counterpart.exists():
            missing.append(f.name)
        elif digest(f) == digest(counterpart):
            same.append(f.name)
        else:
            differ.append(f.name)

    if not a.quiet:
        for n in differ:
            print(f"  DIFFERS  {n}")
        for n in missing:
            print(f"  MISSING  {n}  (no committed counterpart)")
    stamped = sum(1 for n in same if n.lower().endswith((".pdf", ".xlsx")))
    note = f", {stamped} matched after ignoring embedded timestamps" if stamped else ""
    print(f"{len(same)}/{len(files)} identical{note}")
    return 0 if not (differ or missing) else 1


if __name__ == "__main__":
    sys.exit(main())
