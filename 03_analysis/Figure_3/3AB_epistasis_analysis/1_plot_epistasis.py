"""
Whether substitutions add up, once you know what each one does alone.

Each panel puts a multiple mutant's measured kcat(app) against what its parts predict if they
acted independently. Points on the diagonal are additive; the grey band is the twofold
corridor either side of it, so anything outside the band is epistatic by more than a factor
of two. Three panels: doubles from their singles, triples from their singles, and triples from
a double plus the remaining single.

The prediction needs single mutants, and the campaign only scored a few dozen. The 1,122
singles in 01_fitness_score_processing/max_single_mutants.csv come from Max's deep mutational
scan of the same enzyme, scored the same way, and are what make the panels possible; the
multiple mutants are the campaign's own high-confidence table. kcat(app) is computed with the conversion fitted in
01_fitness_score_processing/03_deriving_kcatapp.

epistasis_models.py holds the expectation arithmetic and defines four further panels, for
quadruple mutants, that the figure does not use. Set ALL_PANELS to draw those as well.

Each row of the CSV is labelled positive, negative or additive on the same rule the band
draws: observed above twofold the expectation is positive epistasis, below half of it is
negative, anything inside the corridor is additive.

Outputs, written next to this script:
    epistasis_expectations.csv
    epistasis.svg / .png                  logarithmic axes, as published
    epistasis_linear_axes.svg / .png
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
    FITNESS_HIGH_CONF_CSV,
    KCAT_APP_FIT_JSON,
    MAX_SINGLE_MUTANTS_CSV,
)

from epistasis_models import (  # noqa: E402
    CATEGORY_COLOURS,
    CATEGORY_ORDER,
    PANELS,
    panel_rows,
)

COLUMNS = ["mut_id", "fitness_final", "sigma_final", "Nham_aa"]

ALL_PANELS = False  # True also draws the four quadruple-mutant panels
PANELS_DRAWN = PANELS if ALL_PANELS else PANELS[:3]

BAND = 2.0  # the grey corridor is this factor either side of the diagonal

plt.rcParams["svg.fonttype"] = "none"  # keep SVG text editable
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["font.size"] = 11


def variants():
    """The campaign's scored variants and Max's single mutants, in one table."""
    campaign = pd.read_csv(FITNESS_HIGH_CONF_CSV)[COLUMNS]
    singles = pd.read_csv(MAX_SINGLE_MUTANTS_CSV)[COLUMNS]
    overlap = set(campaign["mut_id"]) & set(singles["mut_id"])
    if overlap:
        raise ValueError(f"{len(overlap)} variants are in both tables, e.g. {sorted(overlap)[:3]}")
    print(f"{len(campaign)} scored campaign variants and {len(singles)} of Max's singles")
    return pd.concat([campaign, singles], ignore_index=True)


def with_kcat_app(table):
    fit = json.loads(KCAT_APP_FIT_JSON.read_text())
    table = table.copy()
    table["kcat_app"] = fit["a"] * np.exp(fit["b"] * table["fitness_final"])
    table["kcat_app_error"] = abs(fit["b"]) * table["kcat_app"] * table["sigma_final"]
    return table, fit["a"]


def epistasis_call(rows):
    """Label each row by where it sits relative to the twofold corridor."""
    observed = rows["observed_kcat_app"]
    expected = rows["expected_kcat_app"]
    return np.select(
        [observed > expected * BAND, observed < expected / BAND],
        ["positive", "negative"],
        default="additive",
    )


def axis_span(rows, logarithmic):
    """Shared limits and the diagonal's corridor, over every panel drawn."""
    low = min(rows["expected_kcat_app"].min(), rows["observed_kcat_app"].min())
    high = max(rows["expected_kcat_app"].max(), rows["observed_kcat_app"].max())
    if logarithmic:
        span = np.logspace(np.log10(low * 0.85), np.log10(high * 1.15), 200)
        return span, span.min(), span.max()
    span = np.linspace(0.0, high * 1.08, 200)
    return span, 0.0, high * 1.08


def plot(rows, logarithmic, stem):
    span, low, high = axis_span(rows, logarithmic)
    figure, axes = plt.subplots(1, len(PANELS_DRAWN),
                                figsize=(4.4 * len(PANELS_DRAWN), 4.6),
                                sharex=True, sharey=True, squeeze=False)

    for ax, (panel, title) in zip(axes[0], PANELS_DRAWN):
        subset = rows[rows["panel"] == panel]
        for category in CATEGORY_ORDER:
            points = subset[subset["category"] == category]
            if points.empty:
                continue
            ax.scatter(points["expected_kcat_app"], points["observed_kcat_app"], s=50,
                       color=CATEGORY_COLOURS[category], alpha=0.8, label=category)
        if logarithmic:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.fill_between(span, span / BAND, span * BAND, color="lightgrey", alpha=0.6)
        ax.plot(span, span, linestyle="--", color="grey", linewidth=1.5)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.grid(False)
        ax.set_title(f"{title}\nn = {len(subset)}", fontsize=11)
        ax.set_xlabel("Expected kcat(app) [s$^{-1}$]")

    axes[0][0].set_ylabel("Observed kcat(app) [s$^{-1}$]")
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=8,
                          color=CATEGORY_COLOURS[c], label=c) for c in CATEGORY_ORDER]
    axes[0][-1].legend(handles=handles, title="Carries", fontsize=9, loc="upper left",
                       bbox_to_anchor=(1.02, 1.0), frameon=False)

    figure.tight_layout()
    for extension in ("svg", "png"):
        figure.savefig(THIS_DIR / f"{stem}.{extension}", dpi=400)
    plt.close(figure)


def main():
    table, wild_type_kcat_app = with_kcat_app(variants())
    rows = panel_rows(table, wild_type_kcat_app)
    rows = rows[np.isfinite(rows["expected_kcat_app"]) & np.isfinite(rows["observed_kcat_app"])]
    rows = rows[rows["panel"].isin([panel for panel, _ in PANELS_DRAWN])].reset_index(drop=True)

    rows["epistasis"] = epistasis_call(rows)

    rows.to_csv(THIS_DIR / "epistasis_expectations.csv", index=False)
    plot(rows, logarithmic=True, stem="epistasis")
    plot(rows, logarithmic=False, stem="epistasis_linear_axes")

    for panel, _ in PANELS_DRAWN:
        subset = rows[rows["panel"] == panel]
        counts = subset["epistasis"].value_counts()
        print(f"{panel}: {len(subset)} variants, "
              f"{counts.get('positive', 0)} positive, "
              f"{counts.get('negative', 0)} negative, "
              f"{counts.get('additive', 0)} additive")


if __name__ == "__main__":
    main()
