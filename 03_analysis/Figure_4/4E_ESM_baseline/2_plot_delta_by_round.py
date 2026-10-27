import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR
while not (REPO_ROOT / ".project-root").exists():
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

import os

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


from variant_rounds import first_round  # noqa: E402

BASE_DIR = str(THIS_DIR)
ESM_CSV = os.path.join(BASE_DIR, "esm2_single_pass_ppl.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "esm2_single_pass_ppl_with_round.csv")
OUTPUT_PDF = os.path.join(BASE_DIR, "esm2_delta_ppl_vs_delta_fitness_by_round.pdf")
OUTPUT_PNG = os.path.join(BASE_DIR, "esm2_delta_ppl_vs_delta_fitness_by_round.png")

ROUND_ORDER = ["r1", "r2", "r3", "r4", "r5", "unassigned"]
ROUND_COLORS = {
    "r1": "#4C78A8",
    "r2": "#F58518",
    "r3": "#54A24B",
    "r4": "#B279A2",
    "r5": "#E45756",
    "unassigned": "#9A9A9A",
}


def load_with_first_round():
    esm_df = pd.read_csv(ESM_CSV)
    earliest = first_round(esm_df["variant_id"])
    esm_df["first_observed_round"] = earliest.to_numpy()
    esm_df["first_observed_round"] = esm_df["first_observed_round"].fillna("unassigned")
    return esm_df


def plot(df):
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 9

    fig, ax = plt.subplots(figsize=(4.05, 3.35))

    for round_name in ROUND_ORDER:
        sub = df[df["first_observed_round"] == round_name]
        if sub.empty:
            continue
        ax.scatter(
            sub["delta_ppl_star_vs_wt"],
            sub["fitness_delta_vs_wt"],
            s=19 if round_name != "unassigned" else 24,
            color=ROUND_COLORS[round_name],
            edgecolor="white",
            linewidth=0.3,
            alpha=0.82 if round_name != "unassigned" else 0.95,
            label=round_name,
        )

    corr_df = df[["delta_ppl_star_vs_wt", "fitness_delta_vs_wt"]].dropna()
    pearson = stats.pearsonr(corr_df["delta_ppl_star_vs_wt"], corr_df["fitness_delta_vs_wt"])
    spearman = stats.spearmanr(corr_df["delta_ppl_star_vs_wt"], corr_df["fitness_delta_vs_wt"])
    ax.text(
        0.03,
        0.97,
        f"n={len(corr_df)}\nPearson r={pearson.statistic:.2f}\nSpearman rho={spearman.statistic:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.3,
    )
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.axvline(0, color="#777777", linewidth=0.8)
    ax.set_xlabel("Delta ESM2 single-pass PPL* vs WT")
    ax.set_ylabel("Delta dETM fitness vs WT")
    ax.grid(True, color="#E7E7E7", linewidth=1)
    ax.tick_params(axis="both", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.legend(title="First observed", frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_PDF)
    fig.savefig(OUTPUT_PNG, dpi=300)
    plt.close(fig)


def main():
    df = load_with_first_round()
    df.to_csv(OUTPUT_CSV, index=False)
    plot(df)
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_PDF}")
    print(f"Wrote {OUTPUT_PNG}")
    print(df["first_observed_round"].value_counts().reindex(ROUND_ORDER, fill_value=0).to_string())


if __name__ == "__main__":
    main()
