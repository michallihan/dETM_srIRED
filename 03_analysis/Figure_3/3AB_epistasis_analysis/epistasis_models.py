"""
Expected kcat(app) for a multiple mutant, built from its parts.

If two substitutions acted independently their fold-changes over the wild type would
multiply, so the expected kcat(app) of a variant decomposed into pieces is

    expected = product(kcat_app of each piece) / a^(number of pieces - 1)

where a is the fitted kcat(app) of the wild type, the fold-change denominator. A variant sits
on the diagonal when its measured kcat(app) matches that, above it when the substitutions help
each other and below it when they get in each other's way.

Most variants can be cut up more than one way. A triple splits into three singles, or into a
double plus the remaining single three different ways; a quadruple has four decompositions.
Where several exist the expectations are averaged and the count is kept, so a point built from
one surviving decomposition can be told from one built from four.

A decomposition only counts if every piece has been measured. The single mutants are what make
that possible at all, and most of them come from Max's dataset rather than the campaign.
"""

import itertools

import numpy as np
import pandas as pd

# Panels, in the order the original laid them out. The first three are the figure; the other
# four are the rest of the decompositions, computed the same way and kept for reference.
PANELS = (
    ("double_from_singles", "Double mutants"),
    ("triple_from_singles", "Triple mutants against a singles model"),
    ("triple_from_double_single", "Triple mutants against a doubles-plus-single model"),
    ("quad_from_singles", "Quadruple mutants against a singles model"),
    ("quad_from_single_single_double", "Quadruple mutants against a singles-plus-double model"),
    ("quad_from_double_double", "Quadruple mutants against a doubles-plus-doubles model"),
    ("quad_from_triple_single", "Quadruple mutants against a triples-plus-single model"),
)

PANEL_TITLES = dict(PANELS)

# Variants are coloured by which of the winning lineage's substitutions they carry.
CATEGORY_COLOURS = {
    "T241A": "#9FC4CF",
    "T241A+S68P": "#EF8C8E",
    "S68P-P203Q-T241A": "#79B86D",
    "S68P-T241A-H244Y": "#79B86D",
    "Other": "#8E8E8E",
}
CATEGORY_ORDER = tuple(CATEGORY_COLOURS)


def substitutions(mut_id):
    return [] if mut_id == "WT" else mut_id.split("-")


def classify(mutations):
    """Which part of the winning lineage a variant carries."""
    carried = set(mutations)
    if {"S68P", "T241A", "H244Y"} <= carried:
        return "S68P-T241A-H244Y"
    if {"S68P", "P203Q", "T241A"} <= carried:
        return "S68P-P203Q-T241A"
    if {"S68P", "T241A"} <= carried:
        return "T241A+S68P"
    if "T241A" in carried:
        return "T241A"
    return "Other"


def _combine(values, wild_type_kcat_app):
    """Multiply the pieces' kcat(app) and divide out the wild type once per extra piece."""
    if any(pd.isna(value) for value in values):
        return np.nan
    return float(np.prod(values)) / wild_type_kcat_app ** (len(values) - 1)


def _average(values):
    finite = [value for value in values if pd.notna(value) and np.isfinite(value)]
    return float(np.mean(finite)) if finite else np.nan


def _decompositions(mutations, order):
    """Every way of cutting a variant up, as lists of piece names, panel by panel."""
    if order == 2:
        yield "double_from_singles", [list(mutations)]
    elif order == 3:
        yield "triple_from_singles", [list(mutations)]
        yield "triple_from_double_single", [
            ["-".join(pair), *[m for m in mutations if m not in pair]]
            for pair in itertools.combinations(mutations, 2)
        ]
    elif order == 4:
        yield "quad_from_singles", [list(mutations)]
        yield "quad_from_single_single_double", [
            ["-".join(pair), *[m for m in mutations if m not in pair]]
            for pair in itertools.combinations(mutations, 2)
        ]
        yield "quad_from_double_double", [
            ["-".join(pair), "-".join(m for m in mutations if m not in pair)]
            for pair in (mutations[:1] + mutations[1:2], mutations[:1] + mutations[2:3],
                         mutations[:1] + mutations[3:4])
        ]
        yield "quad_from_triple_single", [
            ["-".join(m for m in mutations if m != left_out), left_out]
            for left_out in mutations
        ]


def panel_rows(table, wild_type_kcat_app):
    """One row per variant per panel: what was expected, and what was measured."""
    measured = table.set_index("mut_id")["kcat_app"].to_dict()
    rows = []
    for mut_id, observed in zip(table["mut_id"], table["kcat_app"]):
        mutations = substitutions(mut_id)
        category = classify(mutations)
        for panel, cuts in _decompositions(mutations, len(mutations)):
            expectations = [_combine([measured.get(piece, np.nan) for piece in pieces],
                                     wild_type_kcat_app) for pieces in cuts]
            rows.append({
                "panel": panel,
                "title": PANEL_TITLES[panel],
                "mut_id": mut_id,
                "n_mutations": len(mutations),
                "category": category,
                "expected_kcat_app": _average(expectations),
                "observed_kcat_app": observed,
                "n_decompositions": int(sum(pd.notna(value) for value in expectations)),
            })
    return pd.DataFrame(rows)
