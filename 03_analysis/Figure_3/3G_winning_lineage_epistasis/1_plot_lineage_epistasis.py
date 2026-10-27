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
    LRDMS_SINGLE_MUTANTS_CSV,
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
VARIANT_LABEL_SIZE = 6.5  # the names run to four lines, so they set how narrow this can go
Y_LIMIT = 140
Y_STEP = 20

plt.rcParams["pdf.fonttype"] = 42  # embed TrueType so the PDF's text stays editable
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]


def scores():
    """kcat(app) and its error for every variant this panel needs, campaign first.

    The lrDMS singles carry fitness scores only, so theirs are converted here with the same fit.
    """
    columns = ["mut_id", "kcat_app", "kcat_app_error"]
    campaign = pd.read_csv(FITNESS_HIGH_CONF_KCAT_CSV)[columns]

    fit = json.loads(KCAT_APP_FIT_JSON.read_text())
    singles = pd.read_csv(LRDMS_SINGLE_MUTANTS_CSV)
    singles = singles[~singles["mut_id"].isin(set(campaign["mut_id"]))].copy()
    singles["kcat_app"] = fit["a"] * np.exp(fit["b"] * singles["fitness_final"])
    singles["kcat_app_error"] = (abs(fit["b"]) * singles["kcat_app"]
                                 * singles["sigma_final"])

    return pd.concat([campaign, singles[columns]], ignore_index=True).set_index("mut_id")


def wild_type_scale():
    """a: the turnover the fit gives at zero fitness, the denominator fold-changes are over."""
    return json.loads(KCAT_APP_FIT_JSON.read_text())["a"]


def lineage_table():
    measured = scores()
    scale = wild_type_scale()

    rows = []
    for variant, step in LINEAGE.items():
        for name in (variant, *(step or ())):
            if name not in measured.index:
                raise KeyError(f"{name} has no kcat(app)")
        observed = measured.loc[variant]

        row = {"variant": variant, "built_from": "", "expected_kcat_app": np.nan,
               "expected_kcat_app_error": np.nan,
               "observed_kcat_app": observed["kcat_app"],
               "observed_kcat_app_error": observed["kcat_app_error"],
               "observed_over_expected": np.nan}
        if step is not None:
            parent, added = (measured.loc[name] for name in step)
            expected = parent["kcat_app"] * added["kcat_app"] / scale
            relative = np.hypot(parent["kcat_app_error"] / parent["kcat_app"],
                                added["kcat_app_error"] / added["kcat_app"])
            row.update({"built_from": " + ".join(step),
                        "expected_kcat_app": expected,
                        "expected_kcat_app_error": expected * relative,
                        "observed_over_expected": observed["kcat_app"] / expected})
        rows.append(row)
    return pd.DataFrame(rows)


def label(variant):
    """Variant name broken over lines, as the panel prints it."""
    return "-\n".join(variant.split("-"))


def plot(table):
    positions = np.arange(len(table))
    figure, ax = plt.subplots(figsize=(3.7, 3.1))

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
    # tick_params after set_xticklabels would override the size, so it goes first.
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=VARIANT_LABEL_SIZE)
    ax.set_xticks(positions)
    ax.set_xticklabels([label(v) for v in table["variant"]])
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
