import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR
while not (REPO_ROOT / ".project-root").exists():
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from repo_paths import FITNESS_HIGH_CONF_KCAT_CSV  # noqa: E402
from variant_rounds import ROUNDS, variants_per_round  # noqa: E402

SELECTED_FRACTION = 0.001  # the top 0.1% of each round's population was sorted
QUANTILES = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90)
GATE_QUANTILE = 0.10  # the one quoted as the gate proxy

ROUND_COLOURS = ("#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756")

plt.rcParams["svg.fonttype"] = "none"  # keep SVG text editable
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "pdf.fonttype": 42, "ps.fonttype": 42})


def survivors():
    """Every scored mutant that came back from a round, as a fold-change over the wild type."""
    kcat = pd.read_csv(FITNESS_HIGH_CONF_KCAT_CSV)[["mut_id", "kcat_app", "kcat_app_error",
                                                    "Nham_aa"]]
    wild_type = float(kcat.loc[kcat["mut_id"] == "WT", "kcat_app"].iloc[0])

    rounds = variants_per_round()[["round", "round_label", "mut_id"]]
    table = rounds.merge(kcat, on="mut_id", how="inner", validate="many_to_one")
    table["kcat_app_fold_wt"] = table["kcat_app"] / wild_type
    table["log2_kcat_app_fold_wt"] = np.log2(table["kcat_app_fold_wt"])

    table = table[table["mut_id"].ne("WT") & table["kcat_app_fold_wt"].gt(0)]
    return table.reset_index(drop=True), wild_type


def gate_estimates(table, wild_type_kcat_app):
    rows = []
    for round_number, group in table.groupby("round"):
        values = group["kcat_app_fold_wt"]
        quantiles = values.quantile(list(QUANTILES))
        rows.append({
            "round": int(round_number),
            "n_survivors": len(group),
            "selected_fraction_of_population": SELECTED_FRACTION,
            "wild_type_kcat_app": wild_type_kcat_app,
            "gate_min_fold_wt": quantiles.loc[0.0],
            "gate_p01_fold_wt": quantiles.loc[0.01],
            "gate_p05_fold_wt": quantiles.loc[0.05],
            "gate_p10_fold_wt": quantiles.loc[0.10],
            "p25_fold_wt": quantiles.loc[0.25],
            "median_fold_wt": quantiles.loc[0.50],
            "p75_fold_wt": quantiles.loc[0.75],
            "p90_fold_wt": quantiles.loc[0.90],
            "mean_fold_wt": values.mean(),
            "max_fold_wt": values.max(),
        })
    return pd.DataFrame(rows).sort_values("round").reset_index(drop=True)


def save(figure, stem):
    for extension in ("svg", "png"):
        figure.savefig(THIS_DIR / f"{stem}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def gate(estimates, round_number):
    return float(estimates.loc[estimates["round"] == round_number, "gate_p10_fold_wt"].iloc[0])


def plot_gate_position(table, estimates):
    figure, (left, right) = plt.subplots(1, 2, figsize=(8.2, 3.3))

    left.plot(estimates["round"], estimates["median_fold_wt"], marker="o",
              label="Median survivor")
    left.plot(estimates["round"], estimates["gate_p10_fold_wt"], marker="D",
              label=f"p{int(GATE_QUANTILE * 100)} gate estimate")
    left.axhline(1, color="black", lw=0.8)
    left.set_yscale("log")
    left.set_xlabel("Round")
    left.set_ylabel("kcat(app), fold over wild type")
    left.set_title("Where the gate sat")
    left.legend(frameon=False, fontsize=8)

    for colour, round_number in zip(ROUND_COLOURS, ROUNDS):
        values = np.sort(table.loc[table["round"] == round_number, "kcat_app_fold_wt"])
        right.plot(values, np.arange(1, len(values) + 1) / len(values),
                   label=f"r{round_number}", color=colour, lw=1.7)
        right.scatter([gate(estimates, round_number)], [GATE_QUANTILE], s=82, marker="o",
                      facecolor="white", edgecolor=colour, linewidth=2.2, alpha=0.75, zorder=4)
    right.axvline(1, color="black", lw=0.8)
    right.set_xscale("log")
    right.set_xlabel("kcat(app), fold over wild type")
    right.set_ylabel("Share of survivors at or below")
    right.set_title("The lower tail bounds the gate")
    right.legend(frameon=False, fontsize=8, ncol=2)

    figure.suptitle("Gate estimated from the survivors, assuming the top "
                    f"{SELECTED_FRACTION:.1%} of each round was sorted", y=1.04)
    save(figure, "gate_position")


def label_gates(ax, estimates):
    top = ax.get_ylim()[1]
    for colour, round_number in zip(ROUND_COLOURS, ROUNDS):
        value = gate(estimates, round_number)
        ax.axvline(value, color=colour, lw=1.4, ls="--", alpha=0.9)
        ax.text(value, top * (0.92 - 0.08 * (round_number - 1)),
                f"r{round_number} p{int(GATE_QUANTILE * 100)}\n{value:.2f}x", color=colour,
                ha="center", va="top", fontsize=8, rotation=90,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5})


def plot_histograms(table, estimates):
    figure, ax = plt.subplots(figsize=(6.6, 4.0))
    edges = np.logspace(np.log10(max(table["kcat_app_fold_wt"].min() * 0.8, 0.1)),
                        np.log10(table["kcat_app_fold_wt"].max() * 1.15), 34)
    for colour, round_number in zip(ROUND_COLOURS, ROUNDS):
        values = table.loc[table["round"] == round_number, "kcat_app_fold_wt"]
        ax.hist(values, bins=edges, density=True, histtype="step", linewidth=2.0,
                color=colour, label=f"r{round_number}")
    ax.axvline(1, color="black", lw=1.0, label="Wild type")
    label_gates(ax, estimates)
    ax.set_xscale("log")
    ax.set_xlabel("kcat(app), fold over wild type")
    ax.set_ylabel("Density among survivors")
    ax.set_title("Survivor distributions, with the estimated gates")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    save(figure, "kcat_app_distributions_histogram")


def plot_kdes(table, estimates):
    figure, ax = plt.subplots(figsize=(6.6, 4.0))
    folds = np.logspace(np.log10(max(table["kcat_app_fold_wt"].min() * 0.8, 0.1)),
                        np.log10(table["kcat_app_fold_wt"].max() * 1.15), 400)
    for colour, round_number in zip(ROUND_COLOURS, ROUNDS):
        values = table.loc[table["round"] == round_number, "log2_kcat_app_fold_wt"].to_numpy()
        density = gaussian_kde(values)(np.log2(folds))
        ax.plot(folds, density, color=colour, lw=2.2, label=f"r{round_number}")
        ax.fill_between(folds, density, color=colour, alpha=0.08)
    ax.axvline(1, color="black", lw=1.0, label="Wild type")
    label_gates(ax, estimates)
    ax.set_xscale("log")
    ax.set_xlabel("kcat(app), fold over wild type")
    ax.set_ylabel("Density on log2 fold-change")
    ax.set_title("Smoothed survivor distributions, with the estimated gates")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    save(figure, "kcat_app_distributions_kde")


def report(estimates, wild_type_kcat_app):
    def across(column, unit="x"):
        return ", ".join(f"r{int(row['round'])}={row[column]:.2f}{unit}"
                         for _, row in estimates.iterrows())

    lines = [
        "Gate estimates by round, from the kcat(app) of the variants that survived each sort",
        "",
        f"Wild-type kcat(app) = {wild_type_kcat_app:.3f} s-1.",
        f"The top {SELECTED_FRACTION:.1%} of each round's population was sorted.",
        "Only survivors are observed, so the gate is the truncation point below this "
        "distribution. The observed minimum is a lower edge, the 5th percentile a "
        "conservative estimate, and the 10th percentile the practical proxy, being less "
        "dominated by a single low-end outlier.",
        "",
        "Per round, as a fold-change over the wild type:",
    ]
    for _, row in estimates.iterrows():
        lines.append(
            f"r{int(row['round'])}: n={int(row['n_survivors'])}, "
            f"min={row['gate_min_fold_wt']:.2f}x, p5={row['gate_p05_fold_wt']:.2f}x, "
            f"p10={row['gate_p10_fold_wt']:.2f}x, median={row['median_fold_wt']:.2f}x, "
            f"p90={row['p90_fold_wt']:.2f}x"
        )
    lines += [
        "",
        "Reading:",
        f"Conservative p5 estimates: {across('gate_p05_fold_wt')}.",
        f"p10 gate proxies: {across('gate_p10_fold_wt')}.",
        f"Median survivor: {across('median_fold_wt')}.",
        "The lower tail stays below or near the wild type through most rounds, so the "
        "recovered set contains survivors with low kcat(app) despite a nominal top-0.1% "
        "sort. Measurement noise, imperfect genotype recovery at the sorting boundary, and "
        "kcat(app) being inferred from fitness rather than measured per clone all contribute.",
        "The clearer signal is the shift of the whole distribution: the median survivor moves "
        f"from {estimates['median_fold_wt'].iloc[0]:.1f}x the wild type in r1 to "
        f"{estimates['median_fold_wt'].iloc[-1]:.1f}x in r5, and the 90th percentile from "
        f"{estimates['p90_fold_wt'].iloc[0]:.1f}x to {estimates['p90_fold_wt'].iloc[-1]:.1f}x.",
        "",
        "Caveat:",
        "This reads the lower tail of the recovered survivors, not the library that went into "
        "the sort. Survivors too rare to be recovered would bias the apparent gate upward.",
    ]
    (THIS_DIR / "gate_estimates.txt").write_text("\n".join(lines) + "\n")


def main():
    table, wild_type_kcat_app = survivors()
    estimates = gate_estimates(table, wild_type_kcat_app)

    estimates.to_csv(THIS_DIR / "gate_estimates_by_round.csv", index=False)
    table.to_csv(THIS_DIR / "survivors_by_round.csv", index=False)

    plot_gate_position(table, estimates)
    plot_histograms(table, estimates)
    plot_kdes(table, estimates)
    report(estimates, wild_type_kcat_app)

    print(estimates[["round", "n_survivors", "gate_p05_fold_wt", "gate_p10_fold_wt",
                     "median_fold_wt", "p90_fold_wt"]].to_string(index=False))


if __name__ == "__main__":
    main()
