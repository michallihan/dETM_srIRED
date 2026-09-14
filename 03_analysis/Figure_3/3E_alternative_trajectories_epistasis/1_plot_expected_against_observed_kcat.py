"""
Whether the routes that were not taken would have worked, measured one enzyme at a time.

T241A opens three routes: adding S68P, A72T or G37C. Each pair was purified and assayed, and
so were the triples that add S68P on top of the A72T and G37C doubles. That makes it possible
to ask of each one whether its substitutions combine as though they were independent.

Independence means fold-changes over the wild type multiply, so for a variant built from two
measured pieces

    expected kcat = kcat(first piece) * kcat(second piece) / kcat(wild type)

The error on that is propagated from all three measurements. Since the expectation is a
product and a quotient, the relative errors add in quadrature:

    expected_error / expected = sqrt( (dA/A)^2 + (dB/B)^2 + (dW/W)^2 )

The original calculation carried only the wild type's error, which understated the expected
bars - the pieces are measured no better than the variant itself, and on the triples the
double contributes most of the uncertainty. Both are in the csv, the original as
expected_kcat_error_wild_type_only, so the change is visible rather than silent.

The doubles are built from their two singles. The triples are built from the double that
already carries T241A plus S68P on its own - S68P added last, to the established background -
which is the comparison the panel is making and not the only decomposition available.

Measurements come from 04_biophysics/srired_chx_cpa_alternative_paths_kinetics.csv, which is
its own assay series: the wild type turns over at 4.7 s-1 there. The larger biophysics table
holds some of the same variants on a different scale, and the two must not be mixed.

Outputs, written next to this script:
    expected_against_observed_kcat.csv
    expected_against_observed_kcat.svg / .png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR
while not (REPO_ROOT / ".project-root").exists():
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from repo_paths import ALTERNATIVE_PATHS_KINETICS_CSV  # noqa: E402

WILD_TYPE = "WT"

# variant -> the two measured pieces its expectation is built from, in the panel's order.
DECOMPOSITIONS = {
    "S68P-T241A": ("S68P", "T241A"),
    "A72T-T241A": ("A72T", "T241A"),
    "G37C-T241A": ("G37C", "T241A"),
    "S68P-A72T-T241A": ("S68P", "A72T-T241A"),
    "G37C-S68P-T241A": ("S68P", "G37C-T241A"),
}

OBSERVED_COLOUR = "#ABCDC2"
EXPECTED_COLOUR = "#9A9A9A"
BAR_WIDTH = 0.38
Y_LIMIT = 150
Y_STEP = 50

plt.rcParams["svg.fonttype"] = "none"  # keep SVG text editable
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]


def canonical(variant):
    """One spelling for a variant: substitutions in residue order, joined by hyphens."""
    substitutions = [s for s in variant.replace(",", "-").split("-") if s]
    if substitutions == [WILD_TYPE]:
        return WILD_TYPE
    return "-".join(sorted(substitutions, key=lambda s: int(s[1:-1])))


def kinetics():
    """Measured kcat and its error, keyed by canonical variant name."""
    table = pd.read_csv(ALTERNATIVE_PATHS_KINETICS_CSV)
    table["variant"] = table["aa_variant"].map(canonical)
    duplicated = table["variant"][table["variant"].duplicated()]
    if len(duplicated):
        raise ValueError(f"the kinetics table names {list(duplicated)} more than once")
    return table.set_index("variant")[["kcat", "kcat (error) [s-1]"]].rename(
        columns={"kcat (error) [s-1]": "kcat_error"})


def expectations(measured):
    """For each variant, what its pieces predict and what it actually does."""
    wild_type = measured.loc[WILD_TYPE]
    rows = []
    for variant, pieces in DECOMPOSITIONS.items():
        for name in (variant, *pieces):
            if name not in measured.index:
                raise KeyError(f"{name} is not in the kinetics table")
        expected = float(np.prod([measured.loc[p, "kcat"] for p in pieces]) / wild_type["kcat"])
        # A product and a quotient, so the relative errors add in quadrature.
        relative = [measured.loc[p, "kcat_error"] / measured.loc[p, "kcat"] for p in pieces]
        relative.append(wild_type["kcat_error"] / wild_type["kcat"])
        rows.append({
            "variant": variant,
            "built_from": " x ".join(pieces),
            "expected_kcat": expected,
            "expected_kcat_error": expected * float(np.sqrt(np.sum(np.square(relative)))),
            # What the original workbook drew: the wild type's error and nothing else.
            "expected_kcat_error_wild_type_only":
                expected * wild_type["kcat_error"] / wild_type["kcat"],
            "observed_kcat": measured.loc[variant, "kcat"],
            "observed_kcat_error": measured.loc[variant, "kcat_error"],
        })
    table = pd.DataFrame(rows)
    table["observed_over_expected"] = table["observed_kcat"] / table["expected_kcat"]
    return table


def label(variant):
    """Variant name broken over lines, as the panel prints it."""
    return "-\n".join(variant.split("-"))


def plot(table):
    positions = np.arange(len(table))
    figure, ax = plt.subplots(figsize=(4.1, 2.9))

    ax.bar(positions - BAR_WIDTH / 2, table["expected_kcat"], BAR_WIDTH,
           yerr=table["expected_kcat_error"], color=EXPECTED_COLOUR, label="Expected",
           error_kw={"elinewidth": 0.8, "capsize": 2, "capthick": 0.8}, zorder=3)
    ax.bar(positions + BAR_WIDTH / 2, table["observed_kcat"], BAR_WIDTH,
           yerr=table["observed_kcat_error"], color=OBSERVED_COLOUR, label="Observed",
           error_kw={"elinewidth": 0.8, "capsize": 2, "capthick": 0.8}, zorder=3)

    ax.set_ylabel("$k_\\mathregular{cat}$ [s$^{-1}$]", fontsize=11)
    ax.set_ylim(0, Y_LIMIT)
    ax.set_yticks(range(0, Y_LIMIT + 1, Y_STEP))
    ax.set_xticks(positions)
    ax.set_xticklabels([label(v) for v in table["variant"]], fontsize=8.5)
    ax.tick_params(axis="both", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index("Observed"), labels.index("Expected")]
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc="upper left", fontsize=9, frameon=False)

    figure.tight_layout()
    for extension in ("svg", "png"):
        figure.savefig(THIS_DIR / f"expected_against_observed_kcat.{extension}", dpi=400)
    plt.close(figure)


def main():
    table = expectations(kinetics())
    table.to_csv(THIS_DIR / "expected_against_observed_kcat.csv", index=False)
    plot(table)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
