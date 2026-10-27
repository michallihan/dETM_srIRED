import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402
from scipy.stats import linregress, pearsonr  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR
while not (REPO_ROOT / ".project-root").exists():
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from repo_paths import BIOPHYSICAL_DATA_CSV, FITNESS_HIGH_CONF_CSV  # noqa: E402

INITIAL_GUESS = (1.0, 0.1)

plt.rcParams["svg.fonttype"] = "none"  # keep SVG text editable
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["font.size"] = 15


def exponential(fitness, a, b):
    return a * np.exp(b * fitness)


def measured_kcat():
    """The purified variants' kcat, keyed the way the fitness table names variants."""
    measurements = pd.read_csv(BIOPHYSICAL_DATA_CSV)
    # The biophysics table separates substitutions with commas; everything else uses dashes.
    keys = measurements["aa_variant"].str.replace(",", "-", regex=False).str.strip()
    return pd.DataFrame({"mut_id": keys,
                         "kcat_measured": measurements["kcat"],
                         "kcat_measured_error": measurements["kcat (error) [s-1]"]})


def plot(overlap, regression, pearson):
    figure, ax = plt.subplots(figsize=(6, 5))
    ax.errorbar(overlap["kcat_measured"], overlap["kcat_app"], yerr=overlap["kcat_app_error"],
                fmt="o", color="black", ecolor="black", elinewidth=1, capsize=0, markersize=5)

    span = np.linspace(overlap["kcat_measured"].min(), overlap["kcat_measured"].max(), 200)
    ax.plot(span, regression.intercept + regression.slope * span, color="darkred")
    ax.plot(span, span, color="grey", linestyle="--", linewidth=1)

    ax.text(0.05, 0.95,
            f"y = {regression.slope:.2f}x + {regression.intercept:.2f}\n"
            f"R² = {regression.rvalue ** 2:.2f}\nPearson r = {pearson.statistic:.2f}",
            transform=ax.transAxes, verticalalignment="top", horizontalalignment="left")
    ax.set_xlabel("kcat of the purified enzyme [s$^{-1}$]")
    ax.set_ylabel("kcat(app) from the fitness score [s$^{-1}$]")

    figure.tight_layout()
    for extension in ("svg", "png"):
        figure.savefig(THIS_DIR / f"kcat_app_vs_measured_kcat.{extension}", dpi=400)
    plt.close(figure)


def main():
    fitness = pd.read_csv(FITNESS_HIGH_CONF_CSV)
    table = fitness.merge(measured_kcat(), on="mut_id", how="left")
    if len(table) != len(fitness):
        raise ValueError("the biophysics table has more than one row for a scored variant")

    fitted = table[["fitness_final", "kcat_measured"]].dropna()
    fitted = fitted[fitted["kcat_measured"] > 0]
    (a, b), _ = curve_fit(exponential, fitted["fitness_final"].to_numpy(),
                          fitted["kcat_measured"].to_numpy(), p0=INITIAL_GUESS, maxfev=100000)

    predicted = exponential(fitted["fitness_final"], a, b)
    residual = np.sum((fitted["kcat_measured"] - predicted) ** 2)
    total = np.sum((fitted["kcat_measured"] - fitted["kcat_measured"].mean()) ** 2)
    r2 = 1.0 - residual / total

    log_fit = linregress(fitted["fitness_final"], np.log(fitted["kcat_measured"]))
    linear_fit = linregress(fitted["fitness_final"], fitted["kcat_measured"])

    table["kcat_app"] = exponential(table["fitness_final"], a, b)
    table["kcat_app_error"] = abs(b) * table["kcat_app"] * table["sigma_final"]

    columns = [*fitness.columns, "kcat_app", "kcat_app_error"]
    table[columns].to_csv(THIS_DIR / "final_fitness_scores_ETM_with_kcat_app.csv", index=False)

    overlap = table[["mut_id", "kcat_measured", "kcat_app", "kcat_app_error"]].dropna()
    regression = linregress(overlap["kcat_measured"], overlap["kcat_app"])
    pearson = pearsonr(overlap["kcat_measured"], overlap["kcat_app"])
    plot(overlap, regression, pearson)

    (THIS_DIR / "kcat_app_fit.json").write_text(json.dumps(
        {"model": "kcat = a * exp(b * fitness_final)", "a": a, "b": b, "n_overlap": len(fitted)},
        indent=2) + "\n")

    (THIS_DIR / "kcat_app_fit_summary.txt").write_text("\n".join([
        "Fitted model: kcat = a * exp(b * fitness_final)",
        f"n_overlap = {len(fitted)}",
        f"a = {a:.8f}",
        f"b = {b:.8f}",
        f"R2_exponential_original_space = {r2:.8f}",
        f"log_linear_r2 = {log_fit.rvalue ** 2:.8f}",
        f"log_linear_p = {log_fit.pvalue:.8e}",
        f"linear_r2_original_space = {linear_fit.rvalue ** 2:.8f}",
        f"linear_p_original_space = {linear_fit.pvalue:.8e}",
        "",
        "Check of kcat(app) against the purified kcat it was fitted to",
        f"n_check = {len(overlap)}",
        f"check_slope = {regression.slope:.8f}",
        f"check_intercept = {regression.intercept:.8f}",
        f"check_r2 = {regression.rvalue ** 2:.8f}",
        f"check_pearson_r = {pearson.statistic:.8f}",
        f"check_pearson_p = {pearson.pvalue:.8e}",
    ]) + "\n")

    print(f"kcat = {a:.8f} * exp({b:.8f} * fitness_final) from {len(fitted)} purified variants")
    print(f"exponential R2 = {r2:.4f}; check against measured kcat r = {pearson.statistic:.4f}")


if __name__ == "__main__":
    main()
