"""
Whether each step of the winning lineage delivered what the step before it promised.

The campaign's best enzyme was reached one substitution at a time: T241A, then S68P, then
H244Y and P203Q. At each step the variant already in hand and the substitution about to be
added have both been scored, so the step can be predicted before it is taken, and the
prediction compared with what the enzyme actually did.

The null is that a substitution contributes the same amount wherever it is added. Droplet
fitness is a log-scale quantity, so on that scale contributions add:

    expected fitness = fitness(variant so far) + fitness(the substitution on its own)
    expected sigma   = sqrt( sigma(variant so far)^2 + sigma(the substitution)^2 )

and the expectation is converted to an apparent kcat with the same fit the measured variants
use, kcat_app = a * exp(b * fitness), from 01_fitness_score_processing/03_deriving_kcatapp.
Adding fitness is exactly multiplying kcat(app) fold-changes over a, the fitted turnover at
zero fitness -- which is a shade below the wild-type variant's own kcat_app, since the wild
type scores slightly above zero.

Both terms of the error are carried, the variant's and the substitution's. On the H244Y steps
the substitution dominates it: H244Y was scored once, at low confidence, and its sigma is
0.74 against the 0.24 of the variant it is added to.

The wild type and T241A have no expected bar. They are where the lineage starts - there is
nothing before them to predict from.

H244Y is not in the campaign's own tables at all and comes from Max's single-mutant scan,
which is why this panel reads both.

Outputs, written next to this script:
    lineage_epistasis.csv
    lineage_epistasis.pdf / .png
"""

import json
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

from repo_paths import (  # noqa: E402
    FITNESS_HIGH_CONF_KCAT_CSV,
    KCAT_APP_FIT_JSON,
    MAX_SINGLE_MUTANTS_CSV,
)

# The lineage in the order it was walked. None means the variant is a starting point and gets
# no expectation; otherwise (variant already in hand, substitution added to it).
LINEAGE = {
    "WT": None,
    "T241A": None,
    "S68P-T241A": ("T241A", "S68P"),
    "S68P-P203Q-T241A": ("S68P-T241A", "P203Q"),
    "S68P-T241A-H244Y": ("S68P-T241A", "H244Y"),
    "S68P-P203Q-T241A-H244Y": ("S68P-T241A-H244Y", "P203Q"),
}

OBSERVED_COLOUR = "#ABCDC2"
EXPECTED_COLOUR = "#9A9A9A"
BAR_WIDTH = 0.38
Y_LIMIT = 140
Y_STEP = 20

plt.rcParams["pdf.fonttype"] = 42  # embed TrueType so the PDF's text stays editable
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]


def scores():
    """Fitness and its error for every variant this panel needs, campaign first."""
    campaign = pd.read_csv(FITNESS_HIGH_CONF_KCAT_CSV)[["mut_id", "fitness_final",
                                                        "sigma_final"]]
    singles = pd.read_csv(MAX_SINGLE_MUTANTS_CSV)[["mut_id", "fitness_final", "sigma_final"]]
    singles = singles[~singles["mut_id"].isin(set(campaign["mut_id"]))]
    return pd.concat([campaign, singles], ignore_index=True).set_index("mut_id")


def conversion():
    fit = json.loads(KCAT_APP_FIT_JSON.read_text())
    return fit["a"], fit["b"]


def lineage_table():
    measured = scores()
    a, b = conversion()

    def kcat_app(fitness, sigma):
        value = a * np.exp(b * fitness)
        return value, abs(b) * value * sigma

    rows = []
    for variant, step in LINEAGE.items():
        for name in (variant, *(step or ())):
            if name not in measured.index:
                raise KeyError(f"{name} has no fitness score")
        observed, observed_error = kcat_app(*measured.loc[variant])

        row = {"variant": variant, "built_from": "", "expected_kcat_app": np.nan,
               "expected_kcat_app_error": np.nan, "observed_kcat_app": observed,
               "observed_kcat_app_error": observed_error,
               "observed_over_expected": np.nan}
        if step is not None:
            parent, added = step
            fitness = measured.loc[parent, "fitness_final"] + measured.loc[added, "fitness_final"]
            sigma = float(np.hypot(measured.loc[parent, "sigma_final"],
                                   measured.loc[added, "sigma_final"]))
            expected, expected_error = kcat_app(fitness, sigma)
            row.update({"built_from": f"{parent} + {added}",
                        "expected_kcat_app": expected,
                        "expected_kcat_app_error": expected_error,
                        "observed_over_expected": observed / expected})
        rows.append(row)
    return pd.DataFrame(rows)


def label(variant):
    """Variant name broken over lines, as the panel prints it."""
    return "-\n".join(variant.split("-"))


def plot(table):
    positions = np.arange(len(table))
    figure, ax = plt.subplots(figsize=(4.3, 3.1))

    ax.bar(positions - BAR_WIDTH / 2, table["expected_kcat_app"].fillna(0), BAR_WIDTH,
           yerr=table["expected_kcat_app_error"].fillna(0), color=EXPECTED_COLOUR,
           label="Expected", error_kw={"elinewidth": 0.8, "capsize": 2, "capthick": 0.8},
           zorder=3)
    ax.bar(positions + BAR_WIDTH / 2, table["observed_kcat_app"], BAR_WIDTH,
           yerr=table["observed_kcat_app_error"], color=OBSERVED_COLOUR, label="Observed",
           error_kw={"elinewidth": 0.8, "capsize": 2, "capthick": 0.8}, zorder=3)

    ax.set_ylabel("$k_\\mathregular{cat}^\\mathregular{(app)}$ [s$^{-1}$]", fontsize=11)
    ax.set_ylim(0, Y_LIMIT)
    ax.set_yticks(range(0, Y_LIMIT + 1, Y_STEP))
    ax.set_xticks(positions)
    ax.set_xticklabels([label(v) for v in table["variant"]], fontsize=7.5)
    ax.tick_params(axis="both", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index("Observed"), labels.index("Expected")]
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc="upper left", fontsize=9, frameon=False)

    figure.tight_layout()
    for extension in ("pdf", "png"):
        figure.savefig(THIS_DIR / f"lineage_epistasis.{extension}", dpi=400)
    plt.close(figure)


def main():
    table = lineage_table()
    table.to_csv(THIS_DIR / "lineage_epistasis.csv", index=False)
    plot(table)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
