"""
Figure 2A/2C - select variants carrying a given mutation.

Writes a plain list of node names for every variant carrying the requested mutation, so
that group can be selected and coloured in Cytoscape.

Usage:
    python3 3_make_mutation_selection.py T241A          # exact substitution
    python3 3_make_mutation_selection.py 241            # any substitution at position 241
    python3 3_make_mutation_selection.py T241A S68P     # several at once, one file each

Output, written next to this script:
    colour_<spec>.txt    node names, one per line

NOTE ON PROVENANCE. This replaces convert_list_mutation_format.py (Friederike Nintzel),
which is preserved here in function but not in form. That script did not select anything:
it reformatted an already-selected list, reading a hard-coded 'colour_T241_red.csv' that
is not present in the source data and appears to have been exported by hand from
Cytoscape. Since the name formatting it performed is now done directly from the variant
table, the useful part of it is the selection itself, which is what this script does.
Output of the original could not be reproduced or checked against, because neither its
input nor its output survives.
"""

import ast
import re
import sys
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR
while not (REPO_ROOT / ".project-root").exists():
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from repo_paths import COLLAPSED_UNIQUE_VARIANTS_CSV  # noqa: E402

WT_LABEL = "WT"
SUBSTITUTION = re.compile(r"^([A-Z])(\d+)([A-Z*])$")
POSITION = re.compile(r"^(\d+)$")


def variant_name(mutations):
    return "_".join("".join(triplet) for triplet in mutations) or WT_LABEL


def matches(mutations, spec):
    """True if this variant carries the requested substitution, or any change at a position."""
    sub = SUBSTITUTION.match(spec)
    if sub:
        wt, pos, mut = sub.groups()
        return [wt, pos, mut] in [list(m) for m in mutations]

    pos_only = POSITION.match(spec)
    if pos_only:
        return any(m[1] == pos_only.group(1) for m in mutations)

    raise ValueError(f"cannot parse '{spec}' - expected e.g. 'T241A' or '241'")


def main(specs):
    df = pd.read_csv(COLLAPSED_UNIQUE_VARIANTS_CSV)
    parsed = [ast.literal_eval(m) for m in df["aa_mutations"]]
    names = [variant_name(m) for m in parsed]

    for spec in specs:
        selected = [n for n, muts in zip(names, parsed) if matches(muts, spec)]
        out = THIS_DIR / f"colour_{spec}.txt"
        out.write_text("\n".join(selected) + "\n")
        print(f"{spec}: {len(selected):4d} of {len(names)} variants -> {out.name}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__.strip().split("Usage:")[1].split("Output")[0].strip())
    main(sys.argv[1:])
