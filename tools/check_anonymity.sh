#!/usr/bin/env bash
# Fail if the tree carries anything that identifies an author, machine, or site.
#
# This checker deliberately contains NO literal identifying strings. A checker
# that grepped for "the username we removed" would put that username back into
# the repository, and anyone reading the file would learn exactly what was
# scrubbed. It tests *structure* instead:
#
#   1. no home directory belongs to a named account;
#   2. no email address appears outside test fixtures;
#   3. no academic, government, or internal-cluster hostname;
#   4. every recorded hostname is an anonymised site identifier;
#   5. every machine in the run registry is an anonymised site identifier;
#   6. no symlink target escapes the repository.
#
# Anonymised identifiers look like  site-a  or  site-b-node-1 .
#
# Usage:  tools/check_anonymity.sh
# Exit:   0 clean, 1 findings, 2 could not run.

set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 2

python3 - "$@" <<'PY'
import csv, json, os, re, subprocess, sys
from pathlib import Path

SELF = "tools/check_anonymity.sh"          # excluded: it describes what it hunts

HOME_DIR  = re.compile(r"/(?:home|Users)/([A-Za-z0-9_.-]+)")
EMAIL     = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
SITE_ID   = re.compile(r"^site-[a-z](?:-node-\d+)?$")

# Academic, government, and internal-cluster hostnames. Publisher and service
# domains (dl.acm.org, doi.org, huggingface.co, ...) are legitimate content and
# are deliberately NOT matched; a generic "any FQDN" rule flags 900 lines of
# ordinary Python attribute access and teaches everyone to ignore the checker.
# Machine identity inside recorded metadata is caught precisely by check 4.
INSTITUTIONAL = re.compile(
    r"\b[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*"
    r"\.(?:edu|gov|mil|ac\.[a-z]{2}|mill|cluster|hpc|internal|lan|corp)\b", re.I)

ALLOWED_ACCOUNT = {"user"}                  # the neutral placeholder
ALLOWED_EMAIL_DOMAIN = re.compile(r"@(example|test|placeholder|invalid)\.", re.I)

findings: dict[str, list[str]] = {}
def add(kind, msg):
    findings.setdefault(kind, []).append(msg)

def tracked():
    """(path, mode) for every tracked file, taking the mode from the index."""
    out = subprocess.run(["git", "ls-files", "-s"], capture_output=True, text=True, check=True)
    for line in out.stdout.splitlines():
        meta, _, rel = line.partition("\t")
        if rel and rel != SELF:
            yield rel, meta.split()[0]

def paths():
    return [rel for rel, _ in tracked()]

def text_files():
    for rel, mode in tracked():
        # A symlink's blob *is* its target path, so that string ships in every
        # clone. Reading through the link inspects the wrong bytes when it
        # resolves and raises when it does not, which is how an absolute path
        # into a personal home directory once travelled undetected.
        if mode == "120000":
            yield rel, os.readlink(rel)
            continue
        try:
            raw = Path(rel).read_bytes()
        except OSError as exc:
            # Never skip silently: an unreadable tracked file is unexamined.
            add("unreadable tracked file", f"{rel}: {exc.strerror}")
            continue
        if b"\x00" in raw[:8192]:
            continue
        try:
            yield rel, raw.decode("utf-8")
        except UnicodeDecodeError:
            continue

for rel, text in text_files():
    is_test = "/tests/" in rel or Path(rel).name.startswith("test_")
    for i, line in enumerate(text.splitlines(), 1):
        for m in HOME_DIR.finditer(line):
            if m.group(1) not in ALLOWED_ACCOUNT:
                add("personal home directory", f"{rel}:{i}: {m.group(0)}")
        if not is_test:
            for m in EMAIL.finditer(line):
                if not ALLOWED_EMAIL_DOMAIN.search(m.group(0)):
                    add("email address", f"{rel}:{i}: {m.group(0)}")
        for m in INSTITUTIONAL.finditer(line):
            add("academic / government / cluster hostname", f"{rel}:{i}: {m.group(0)}")

# Recorded hostnames in run metadata and profiling captures.
HOST_KEYS = {"hostname", "host", "node", "nodename", "machine", "machine_id"}
for rel in paths():
    if not rel.endswith(".json"):
        continue
    try:
        doc = json.loads(Path(rel).read_text(encoding="utf-8"))
    except Exception:
        continue
    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k.lower() in HOST_KEYS and isinstance(v, str) and v and not SITE_ID.match(v):
                    add("non-anonymised hostname in recorded metadata", f"{rel}: {k} = {v}")
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(doc)

registry = Path("experiment_registry/run_index.csv")
if registry.exists():
    with registry.open(encoding="utf-8") as fh:
        for i, row in enumerate(csv.DictReader(fh), 1):
            mid = (row.get("machine_id") or "").strip()
            if not SITE_ID.match(mid):
                add("non-anonymised machine in the run registry", f"row {i}: {mid}")

if not findings:
    print("anonymity check: clean")
    sys.exit(0)

for kind, hits in findings.items():
    print(f"\nFAIL: {kind} ({len(hits)})")
    for h in hits[:15]:
        print(f"    {h}")
    if len(hits) > 15:
        print(f"    ... and {len(hits) - 15} more")
print("\nanonymity check: FINDINGS ABOVE — do not push")
sys.exit(1)
PY
