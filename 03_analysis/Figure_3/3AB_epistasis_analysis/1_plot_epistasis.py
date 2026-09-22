"""
Whether substitutions add up, once you know what each one does alone.

Each panel puts a multiple mutant's measured kcat(app) against what its parts predict if they
acted independently. Points on the diagonal are additive; the grey band is the twofold
corridor either side of it, so anything outside the band is epistatic by more than a factor
of two.

Two figures. The first is the published one, three panels: doubles from their singles, triples
from their singles, and triples from a double plus the remaining single. The second takes
every model from triples upward -- two for triples, four for quadruples, six for five-mutation
variants -- on one shared pair of axes, so the models can be compared with each other. A model
with no variant whose pieces were all measured has nothing to draw and is left out; in this
dataset that is the five-mutation triple-plus-double model.

The prediction needs single mutants, and the campaign only scored a few dozen. The 1,122
singles in 01_fitness_score_processing/lrdms_single_mutants.csv come from the lrDMS paper's
deep mutational scan of the same enzyme, scored the same way, and are what make the panels
possible; the multiple mutants are the campaign's own high-confidence table. kcat(app) is
computed with the conversion fitted in 01_fitness_score_processing/03_deriving_kcatapp.

epistasis_models.py holds the expectation arithmetic for every panel.

Each row of the CSV is labelled positive, negative or additive on the same rule the band
draws: observed above twofold the expectation is positive epistasis, below half of it is
negative, anything inside the corridor is additive.

Outputs, written next to this script:
    epistasis_expectations.csv                     every panel of both figures, one row per
                                                   variant per model
    epistasis.svg / .png                           logarithmic axes, as published
    epistasis_linear_axes.svg / .png
    epistasis_higher_order.svg / .png              triples, quadruples, five-mutation variants
    epistasis_higher_order_linear_axes.svg / .png
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
    LRDMS_SINGLE_MUTANTS_CSV,
)

from epistasis_models import (  # noqa: E402
    CATEGORY_COLOURS,
    CATEGORY_ORDER,
    PANELS,
    panel_rows,
)

COLUMNS = ["mut_id", "fitness_final", "sigma_final", "Nham_aa"]

PUBLISHED_PANELS = PANELS[:3]
HIGHER_ORDER_PANELS = PANELS[1:]  # every model from triples upward
HIGHER_ORDER_COLUMNS = 4

BAND = 2.0  # the grey corridor is this factor either side of the diagonal

plt.rcParams["svg.fonttype"] = "none"  # keep SVG text editable
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["font.size"] = 11


def variants():
    """The campaign's scored variants and the lrDMS single mutants, in one table."""
    campaign = pd.read_csv(FITNESS_HIGH_CONF_CSV)[COLUMNS]
    singles = pd.read_csv(LRDMS_SINGLE_MUTANTS_CSV)[COLUMNS]
    overlap = set(campaign["mut_id"]) & set(singles["mut_id"])
    if overlap:
        raise ValueError(f"{len(overlap)} variants are in both tables, e.g. {sorted(overlap)[:3]}")
    print(f"{len(campaign)} scored campaign variants and {len(singles)} of the lrDMS singles")
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


def plot(rows, panels, logarithmic, stem, columns=None):
    """One scatter per panel, all on the same axes, legend in the space after the last."""
    rows = rows[rows["panel"].isin([panel for panel, _ in panels])]
    columns = columns or len(panels)
    grid_rows = -(-len(panels) // columns)
    legend_outside = grid_rows * columns == len(panels)
    span, low, high = axis_span(rows, logarithmic)
    figure, axes = plt.subplots(grid_rows, columns,
                                figsize=(4.4 * columns, 4.6 * grid_rows),
                                sharex=True, sharey=True, squeeze=False)
    axes = axes.flatten()

    for ax, (panel, title) in zip(axes, panels):
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
        if columns != len(panels):  # grid titles are too long for one line
            title = title.replace(" against ", "\nagainst ")
        ax.set_title(f"{title}\nn = {len(subset)}", fontsize=11)
        ax.set_xlabel("Expected kcat(app) [s$^{-1}$]")
        ax.tick_params(labelbottom=True, labelleft=True)

    for ax, (panel, _) in zip(axes, panels):
        if columns == len(panels):
            if ax is axes[0]:
                ax.set_ylabel("Observed kcat(app) [s$^{-1}$]")
        else:
            order = int(rows.loc[rows["panel"] == panel, "n_mutations"].iloc[0])
            ax.set_ylabel(f"Observed {order}-mutation kcat(app) [s$^{{-1}}$]")

    for ax in axes[len(panels):]:
        ax.axis("off")

    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=8,
                          color=CATEGORY_COLOURS[c], label=c) for c in CATEGORY_ORDER]
    if legend_outside:
        axes[len(panels) - 1].legend(handles=handles, title="Carries", fontsize=9,
                                     loc="upper left", bbox_to_anchor=(1.02, 1.0),
                                     frameon=False)
    else:
        axes[len(panels)].legend(handles=handles, title="Carries", fontsize=9,
                                 loc="upper left", frameon=False)

    figure.tight_layout()
    for extension in ("svg", "png"):
        figure.savefig(THIS_DIR / f"{stem}.{extension}", dpi=400)
    plt.close(figure)


def main():
    table, wild_type_kcat_app = with_kcat_app(variants())
    rows = panel_rows(table, wild_type_kcat_app)
    rows = rows[np.isfinite(rows["expected_kcat_app"]) & np.isfinite(rows["observed_kcat_app"])]
    rows = rows.reset_index(drop=True)
    rows["epistasis"] = epistasis_call(rows)
    rows.to_csv(THIS_DIR / "epistasis_expectations.csv", index=False)

    populated = set(rows["panel"])
    higher_order = [(panel, title) for panel, title in HIGHER_ORDER_PANELS if panel in populated]
    for panel, _ in HIGHER_ORDER_PANELS:
        if panel not in populated:
            print(f"{panel}: no variant has all its pieces measured, so it is not drawn")

    plot(rows, PUBLISHED_PANELS, logarithmic=True, stem="epistasis")
    plot(rows, PUBLISHED_PANELS, logarithmic=False, stem="epistasis_linear_axes")
    plot(rows, higher_order, logarithmic=True, stem="epistasis_higher_order",
         columns=HIGHER_ORDER_COLUMNS)
    plot(rows, higher_order, logarithmic=False, stem="epistasis_higher_order_linear_axes",
         columns=HIGHER_ORDER_COLUMNS)

    for panel, _ in PANELS:
        subset = rows[rows["panel"] == panel]
        if subset.empty:
            continue
        counts = subset["epistasis"].value_counts()
        print(f"{panel}: {len(subset)} variants, "
              f"{counts.get('positive', 0)} positive, "
              f"{counts.get('negative', 0)} negative, "
              f"{counts.get('additive', 0)} additive")


if __name__ == "__main__":
    main()
