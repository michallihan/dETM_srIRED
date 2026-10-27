import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR
while not (REPO_ROOT / ".project-root").exists():
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

import math
import os

import esm
import matplotlib.pyplot as plt
import pandas as pd
import torch
from scipy import stats


from repo_paths import FITNESS_HIGH_CONF_KCAT_CSV  # noqa: E402

BASE_DIR = str(THIS_DIR)
INPUT_CSV = FITNESS_HIGH_CONF_KCAT_CSV
OUTPUT_CSV = os.path.join(BASE_DIR, "esm2_single_pass_ppl.csv")
OUTPUT_SUMMARY_CSV = os.path.join(BASE_DIR, "esm2_ppl_correlations.csv")
OUTPUT_PDF = os.path.join(BASE_DIR, "esm2_ppl_vs_fitness.pdf")
OUTPUT_PNG = os.path.join(BASE_DIR, "esm2_ppl_vs_fitness.png")
OUTPUT_DELTA_PDF = os.path.join(BASE_DIR, "esm2_delta_ppl_vs_delta_fitness.pdf")
OUTPUT_DELTA_PNG = os.path.join(BASE_DIR, "esm2_delta_ppl_vs_delta_fitness.png")

MODEL_NAME = "esm2_t33_650M_UR50D"
BATCH_SIZE = 8


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(device):
    model_loader = getattr(esm.pretrained, MODEL_NAME)
    model, alphabet = model_loader()
    model.eval()
    model.to(device)
    return model, alphabet


def score_batch(model, alphabet, device, batch_rows):
    batch_converter = alphabet.get_batch_converter()
    batch = [(row["variant_id"], row["sequence"]) for row in batch_rows]
    labels, _, tokens = batch_converter(batch)
    tokens = tokens.to(device)

    with torch.no_grad():
        logits = model(tokens, repr_layers=[], return_contacts=False)["logits"]
        log_probs = torch.log_softmax(logits, dim=-1)

    rows = []
    for idx, row in enumerate(batch_rows):
        seq_len = len(row["sequence"])
        residue_tokens = tokens[idx, 1 : seq_len + 1]
        residue_log_probs = log_probs[idx, 1 : seq_len + 1, :].gather(1, residue_tokens[:, None]).squeeze(1)
        mean_log_prob = float(residue_log_probs.mean().detach().cpu().item())
        rows.append(
            {
                **row,
                "esm2_model": MODEL_NAME,
                "sequence_length": seq_len,
                "single_pass_mean_log_probability": mean_log_prob,
                "single_pass_negative_mean_log_probability": -mean_log_prob,
                "single_pass_ppl_star": math.exp(-mean_log_prob),
            }
        )
    return rows


def correlation_rows(df):
    pairs = [
        ("single_pass_ppl_star", "fitness_final"),
        ("single_pass_mean_log_probability", "fitness_final"),
        ("single_pass_negative_mean_log_probability", "fitness_final"),
        ("delta_ppl_star_vs_wt", "fitness_delta_vs_wt"),
        ("single_pass_ppl_star", "kcat_app"),
        ("single_pass_mean_log_probability", "kcat_app"),
        ("single_pass_negative_mean_log_probability", "kcat_app"),
        ("delta_ppl_star_vs_wt", "delta_kcat_app_vs_wt"),
    ]
    rows = []
    for x_col, y_col in pairs:
        sub = df[[x_col, y_col]].dropna()
        pearson = stats.pearsonr(sub[x_col], sub[y_col])
        spearman = stats.spearmanr(sub[x_col], sub[y_col])
        rows.append(
            {
                "x": x_col,
                "y": y_col,
                "n": len(sub),
                "pearson_r": pearson.statistic,
                "pearson_p": pearson.pvalue,
                "spearman_rho": spearman.statistic,
                "spearman_p": spearman.pvalue,
            }
        )
    return pd.DataFrame(rows)


def plot_scatter(df, summary):
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 9

    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.scatter(
        df["single_pass_ppl_star"],
        df["fitness_final"],
        s=18,
        color="#A8D0C5",
        edgecolor="white",
        linewidth=0.3,
        alpha=0.8,
    )
    corr = summary[(summary["x"] == "single_pass_ppl_star") & (summary["y"] == "fitness_final")].iloc[0]
    ax.text(
        0.03,
        0.97,
        f"n={int(corr['n'])}\nPearson r={corr['pearson_r']:.2f}\nSpearman rho={corr['spearman_rho']:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
    )
    ax.set_xlabel("ESM2 single-pass PPL*")
    ax.set_ylabel("dETM fitness")
    ax.grid(True, color="#E7E7E7", linewidth=1)
    ax.tick_params(axis="both", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)
    fig.tight_layout()
    fig.savefig(OUTPUT_PDF)
    fig.savefig(OUTPUT_PNG, dpi=300)
    plt.close(fig)


def plot_delta_scatter(df, summary):
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 9

    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.scatter(
        df["delta_ppl_star_vs_wt"],
        df["fitness_delta_vs_wt"],
        s=18,
        color="#A8D0C5",
        edgecolor="white",
        linewidth=0.3,
        alpha=0.8,
    )
    corr = summary[(summary["x"] == "delta_ppl_star_vs_wt") & (summary["y"] == "fitness_delta_vs_wt")].iloc[0]
    ax.text(
        0.03,
        0.97,
        f"n={int(corr['n'])}\nPearson r={corr['pearson_r']:.2f}\nSpearman rho={corr['spearman_rho']:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
    )
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.axvline(0, color="#777777", linewidth=0.8)
    ax.set_xlabel("ΔESM2 single-pass PPL* vs WT")
    ax.set_ylabel("ΔdETM fitness vs WT")
    ax.grid(True, color="#E7E7E7", linewidth=1)
    ax.tick_params(axis="both", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)
    fig.tight_layout()
    fig.savefig(OUTPUT_DELTA_PDF)
    fig.savefig(OUTPUT_DELTA_PNG, dpi=300)
    plt.close(fig)


def main():
    input_df = pd.read_csv(INPUT_CSV)
    rows = []
    for _, row in input_df.iterrows():
        variant_id = row["mut_id"]
        rows.append(
            {
                "variant_id": variant_id,
                "sequence": str(row["aa_seq"]).rstrip("*"),
                "fitness_final": row["fitness_final"],
                "sigma_final": row["sigma_final"],
                "kcat_app": row["kcat_app"],
                "kcat_app_error": row["kcat_app_error"],
                "Nham_aa": row["Nham_aa"],
            }
        )

    device = choose_device()
    print(f"Using device: {device}")
    model, alphabet = load_model(device)

    scored = []
    for start in range(0, len(rows), BATCH_SIZE):
        batch_rows = rows[start : start + BATCH_SIZE]
        scored.extend(score_batch(model, alphabet, device, batch_rows))
        print(f"Scored {min(start + BATCH_SIZE, len(rows))}/{len(rows)}")

    out = pd.DataFrame(scored)
    wt_ppl = float(out.loc[out["variant_id"] == "WT", "single_pass_ppl_star"].iloc[0])
    wt_fitness = float(out.loc[out["variant_id"] == "WT", "fitness_final"].iloc[0])
    wt_kcat = float(out.loc[out["variant_id"] == "WT", "kcat_app"].iloc[0])
    out["ppl_star_over_wt"] = out["single_pass_ppl_star"] / wt_ppl
    out["delta_ppl_star_vs_wt"] = out["single_pass_ppl_star"] - wt_ppl
    out["fitness_delta_vs_wt"] = out["fitness_final"] - wt_fitness
    out["kcat_app_over_wt"] = out["kcat_app"] / wt_kcat
    out["delta_kcat_app_vs_wt"] = out["kcat_app"] - wt_kcat

    summary = correlation_rows(out)
    out.to_csv(OUTPUT_CSV, index=False)
    summary.to_csv(OUTPUT_SUMMARY_CSV, index=False)
    plot_scatter(out, summary)
    plot_delta_scatter(out, summary)

    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_SUMMARY_CSV}")
    print(f"Wrote {OUTPUT_PDF}")
    print(f"Wrote {OUTPUT_PNG}")
    print(f"Wrote {OUTPUT_DELTA_PDF}")
    print(f"Wrote {OUTPUT_DELTA_PNG}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
