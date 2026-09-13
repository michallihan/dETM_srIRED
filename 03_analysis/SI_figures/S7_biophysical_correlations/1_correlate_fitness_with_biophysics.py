"""
Droplet fitness score against what the purified enzymes actually do.

The fitness score is a sorting readout on a variant inside a droplet; the parameters here
were measured on protein expressed and purified one variant at a time. One panel per
parameter, each an exponential fit of the parameter against fitness with a 95% confidence
band, and R² and the fit's standard errors written in the corner.

Twenty-seven variants have both a fitness score and Michaelis-Menten kinetics, which is what
kcat, KM and kcat/KM are fitted on. The melting-temperature panel has 17: fewer variants were
measured, and the fit drops any variant whose ΔTM is not positive, which costs it five more.
That filter is the original's and is kept, but it means the ΔTM panel is fitted on the
stabilising half of the measured set and its R² should be read with that in mind.

Variants are keyed by mutation identifier. The biophysics table separates mutations with
commas where the fitness table uses dashes, so both sides are normalised before the join.
The join is a left join from the biophysics table, so a measured variant with no fitness
score drops out of a panel rather than being silently matched to something else; 44 of the
99 measured variants carry a score.

Outputs, written next to this script:
    biophysics_vs_fitness.csv
    fitness_vs_<parameter>.pdf / .png   for each parameter
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR
while not (REPO_ROOT / ".project-root").exists():
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from repo_paths import BIOPHYSICAL_DATA_CSV, FITNESS_HIGH_CONF_CSV  # noqa: E402

PARAMETERS = ("delta_TM", "kcat", "KM", "kcat_KM")
FITNESS_COLUMN = "fitness_final"

# Axis labels as the published panels carry them. Note that the data file's own error
# columns are headed "KM (error) [mM]" and "kcat/KM (error) [s-1 mM-1]", so the micromolar
# here disagrees with the source table; left as published rather than changed silently.
AXIS_LABELS = {
    "delta_TM": "ΔTM [°C]",
    "kcat": "kcat in purified enzyme [s-1]",
    "KM": "KM in purified enzyme [µM]",
    "kcat_KM": "kcat/KM [s-1 µM-1]",
}

POINT_COLOUR = "black"
FIT_COLOUR = "darkred"
BAND_COLOUR = "red"
CONFIDENCE = 1.96  # 95%, normal approximation
INITIAL_GUESS = (1.0, 0.1)


def exponential(x, a, b):
    """The model fitted to every panel."""
    return a * np.exp(b * x)


def normalise(identifiers):
    """One spelling of a mutation identifier, so the two tables join."""
    return identifiers.astype(str).str.replace(",", "-", regex=False).str.strip()


def load():
    """Measured parameters with each variant's fitness score attached, where it has one."""
    biophysics = pd.read_csv(BIOPHYSICAL_DATA_CSV)
    fitness = pd.read_csv(FITNESS_HIGH_CONF_CSV)

    biophysics["variant"] = normalise(biophysics["aa_variant"])
    fitness["variant"] = normalise(fitness["mut_id"])

    merged = biophysics.merge(fitness[["variant", FITNESS_COLUMN]], on="variant", how="left")
    return merged[["variant", FITNESS_COLUMN, *PARAMETERS]]


def fit_panel(table, parameter):
    """Points, fit and confidence band for one parameter. Returns None if it cannot be fitted."""
    pair = table[[FITNESS_COLUMN, parameter]].dropna()
    measured = len(pair)
    pair = pair[pair[parameter] > 0]
    if len(pair) < 2:
        return None

    x, y = pair[FITNESS_COLUMN].to_numpy(), pair[parameter].to_numpy()
    try:
        popt, pcov = curve_fit(exponential, x, y, p0=INITIAL_GUESS)
    except RuntimeError:
        return None

    predicted = exponential(x, *popt)
    r_squared = 1 - np.sum((y - predicted) ** 2) / np.sum((y - np.mean(y)) ** 2)
    errors = np.sqrt(np.diag(pcov))

    grid = np.linspace(x.min(), x.max(), 200)
    curve = exponential(grid, *popt)
    # Propagate the fit covariance through the model's gradient to get the band.
    spread = np.array([
        np.sqrt(np.array([np.exp(popt[1] * v), popt[0] * v * np.exp(popt[1] * v)])
                @ pcov @ np.array([np.exp(popt[1] * v), popt[0] * v * np.exp(popt[1] * v)]))
        for v in grid
    ])
    return {
        "x": x, "y": y, "grid": grid, "curve": curve, "spread": spread,
        "popt": popt, "errors": errors, "r_squared": r_squared,
        "measured": measured, "fitted": len(pair),
    }


def plot_panel(parameter, fit):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 15,
        "text.color": "black",
        "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(fit["x"], fit["y"], color=POINT_COLOUR)
    ax.plot(fit["grid"], fit["curve"], color=FIT_COLOUR)
    ax.fill_between(fit["grid"],
                    fit["curve"] - CONFIDENCE * fit["spread"],
                    fit["curve"] + CONFIDENCE * fit["spread"],
                    color=BAND_COLOUR, alpha=0.2)

    ax.text(0.05, 0.95,
            "R² = {:.2f}\nσ(a)={:.3f}, σ(b)={:.3f}".format(fit["r_squared"], *fit["errors"]),
            transform=ax.transAxes, verticalalignment="top", horizontalalignment="left",
            color="black")

    ax.set_xlabel("Fitness score in droplets", color="black")
    ax.set_ylabel(AXIS_LABELS.get(parameter, parameter), color="black")

    fig.tight_layout()
    for extension in ("pdf", "png"):
        fig.savefig(THIS_DIR / f"fitness_vs_{parameter}.{extension}", dpi=400)
    plt.close(fig)


def main():
    table = load()
    scored = int(table[FITNESS_COLUMN].notna().sum())
    print(f"{len(table)} measured variants, {scored} of them with a fitness score")
    table.to_csv(THIS_DIR / "biophysics_vs_fitness.csv", index=False)

    print(f"\n{'parameter':10s} {'measured':>9} {'fitted':>7} {'R2':>6} {'a':>8} {'b':>7} "
          f"{'sigma(a)':>9} {'sigma(b)':>9}")
    for parameter in PARAMETERS:
        fit = fit_panel(table, parameter)
        if fit is None:
            print(f"{parameter:10s} could not be fitted")
            continue
        plot_panel(parameter, fit)
        dropped = fit["measured"] - fit["fitted"]
        note = f"   ({dropped} not positive, left out)" if dropped else ""
        print(f"{parameter:10s} {fit['measured']:>9} {fit['fitted']:>7} {fit['r_squared']:6.2f} "
              f"{fit['popt'][0]:8.3f} {fit['popt'][1]:7.3f} "
              f"{fit['errors'][0]:9.3f} {fit['errors'][1]:9.3f}{note}")


if __name__ == "__main__":
    raise SystemExit(main())
