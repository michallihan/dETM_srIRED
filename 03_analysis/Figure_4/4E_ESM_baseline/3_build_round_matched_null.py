"""
The same question asked of variants the campaign never chose.

A variant's ESM2 PPL* depends on how many substitutions it carries and where they sit, so a
drift in PPL* across rounds could be nothing more than variants getting more mutated. The null
controls for that: for each scored variant it builds several random variants carrying the same
number of substitutions, drawn from the pool of substitutions seen in that same round, and
scores them the same way. Whatever separates the observed curve from the null is what
selection did beyond accumulating mutations.

Five replicates from a fixed seed, so the null is reproducible. Needs torch and fair-esm, and
2_ must have run first. Scoring is resumable: an existing null table is reused for rows it
already covers.

Outputs, written next to this script:
    round_matched_null_ppl.csv
    round_matched_null_summary.csv
    esm2_ppl_observed_vs_null.pdf / .png
    esm2_delta_ppl_observed_vs_null.pdf / .png
"""

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR
while not (REPO_ROOT / ".project-root").exists():
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

import math
import os
import random
import re

import esm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.stats import gaussian_kde


from repo_paths import FITNESS_HIGH_CONF_KCAT_CSV  # noqa: E402
from variant_rounds import variants_per_round  # noqa: E402

BASE_DIR = str(THIS_DIR)
OBSERVED_CSV = os.path.join(BASE_DIR, "esm2_single_pass_ppl_with_round.csv")
FITNESS_CSV = FITNESS_HIGH_CONF_KCAT_CSV
OUTPUT_NULL_CSV = os.path.join(BASE_DIR, "round_matched_null_ppl.csv")
OUTPUT_SUMMARY_CSV = os.path.join(BASE_DIR, "round_matched_null_summary.csv")
OUTPUT_PDF = os.path.join(BASE_DIR, "esm2_ppl_observed_vs_null.pdf")
OUTPUT_PNG = os.path.join(BASE_DIR, "esm2_ppl_observed_vs_null.png")
OUTPUT_DELTA_PDF = os.path.join(BASE_DIR, "esm2_delta_ppl_observed_vs_null.pdf")
OUTPUT_DELTA_PNG = os.path.join(BASE_DIR, "esm2_delta_ppl_observed_vs_null.png")

MODEL_NAME = "esm2_t33_650M_UR50D"
BATCH_SIZE = 8
N_NULL_REPLICATES = 5
RANDOM_SEED = 41
ROUND_ORDER = ["r1", "r2", "r3", "r4", "r5"]
ROUND_COLORS = {
    "r1": "#4C78A8",
    "r2": "#F58518",
    "r3": "#54A24B",
    "r4": "#B279A2",
    "r5": "#E45756",
}


def split_mutations(mut_id):
    if pd.isna(mut_id) or mut_id == "WT":
        return []
    return str(mut_id).split("-")


def parse_mutation(mutation):
    match = re.fullmatch(r"([A-Z])(\d+)([A-Z])", mutation)
    if not match:
        return None
    return match.group(1), int(match.group(2)), match.group(3)


def apply_mutations(wt_sequence, mutations):
    seq = list(wt_sequence)
    used_positions = set()
    for mutation in mutations:
        parsed = parse_mutation(mutation)
        if parsed is None:
            return None
        wt_aa, pos, new_aa = parsed
        idx = pos - 1
        if idx < 0 or idx >= len(seq):
            return None
        if seq[idx] != wt_aa or pos in used_positions:
            return None
        seq[idx] = new_aa
        used_positions.add(pos)
    return "".join(seq)


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
    _, _, tokens = batch_converter([(row["variant_id"], row["sequence"]) for row in batch_rows])
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
                "single_pass_mean_log_probability": mean_log_prob,
                "single_pass_negative_mean_log_probability": -mean_log_prob,
                "single_pass_ppl_star": math.exp(-mean_log_prob),
            }
        )
    return rows


def build_round_matched_null():
    rng = random.Random(RANDOM_SEED)
    observed = pd.read_csv(OBSERVED_CSV)
    round_df = variants_per_round().rename(columns={"round_label": "round"})
    fitness_df = pd.read_csv(FITNESS_CSV)
    wt_sequence = str(fitness_df.loc[fitness_df["mut_id"] == "WT", "aa_seq"].iloc[0]).rstrip("*")

    round_variant_sets = {
        round_name: set(round_df.loc[round_df["round"] == round_name, "mut_id"].dropna())
        for round_name in ROUND_ORDER
    }
    round_mutation_pools = {}
    for round_name, variants in round_variant_sets.items():
        pool = []
        for mut_id in variants:
            pool.extend(split_mutations(mut_id))
        parsed_pool = [mutation for mutation in sorted(set(pool)) if parse_mutation(mutation) is not None]
        round_mutation_pools[round_name] = parsed_pool

    rows = []
    observed = observed[observed["first_observed_round"].isin(ROUND_ORDER)].copy()
    for replicate in range(1, N_NULL_REPLICATES + 1):
        for _, obs in observed.iterrows():
            round_name = obs["first_observed_round"]
            n_mutations = int(obs["Nham_aa"])
            pool = round_mutation_pools[round_name]

            if n_mutations == 0:
                mutations = []
                sequence = wt_sequence
            else:
                sequence = None
                for _ in range(500):
                    sample = rng.sample(pool, n_mutations)
                    positions = [parse_mutation(mutation)[1] for mutation in sample]
                    if len(set(positions)) != len(positions):
                        continue
                    sequence = apply_mutations(wt_sequence, sample)
                    if sequence is not None:
                        mutations = sorted(sample, key=lambda mutation: parse_mutation(mutation)[1])
                        break
                if sequence is None:
                    raise RuntimeError(f"Could not sample valid null for {round_name}, n={n_mutations}")

            rows.append(
                {
                    "variant_id": f"null_rep{replicate}_{obs['variant_id']}",
                    "matched_observed_variant_id": obs["variant_id"],
                    "null_replicate": replicate,
                    "first_observed_round": round_name,
                    "Nham_aa": n_mutations,
                    "mutations": "-".join(mutations) if mutations else "WT",
                    "sequence": sequence,
                }
            )

    return rows


def score_rows(rows):
    if os.path.exists(OUTPUT_NULL_CSV):
        existing = pd.read_csv(OUTPUT_NULL_CSV)
        scored_ids = set(existing["variant_id"])
        rows = [row for row in rows if row["variant_id"] not in scored_ids]
        scored = existing.to_dict("records")
        print(f"Resuming from {len(scored)} existing null scores", flush=True)
    else:
        scored = []

    if not rows:
        return pd.DataFrame(scored)

    device = choose_device()
    print(f"Using device: {device}", flush=True)
    model, alphabet = load_model(device)

    total_rows = len(scored) + len(rows)
    for start in range(0, len(rows), BATCH_SIZE):
        batch_rows = rows[start : start + BATCH_SIZE]
        scored.extend(score_batch(model, alphabet, device, batch_rows))
        pd.DataFrame(scored).to_csv(OUTPUT_NULL_CSV, index=False)
        print(f"Scored null {len(scored)}/{total_rows}", flush=True)
    return pd.DataFrame(scored)


def summarize(observed, null):
    rows = []
    for round_name in ROUND_ORDER:
        obs = observed.loc[observed["first_observed_round"] == round_name, "single_pass_ppl_star"].dropna()
        nul = null.loc[null["first_observed_round"] == round_name, "single_pass_ppl_star"].dropna()
        test = stats.mannwhitneyu(obs, nul, alternative="two-sided")
        rows.append(
            {
                "round": round_name,
                "observed_n": len(obs),
                "null_n": len(nul),
                "observed_median_ppl_star": obs.median(),
                "null_median_ppl_star": nul.median(),
                "observed_mean_ppl_star": obs.mean(),
                "null_mean_ppl_star": nul.mean(),
                "median_delta_observed_minus_null": obs.median() - nul.median(),
                "mann_whitney_u_p": test.pvalue,
            }
        )
    return pd.DataFrame(rows)


def plot_overlay(observed, null, value_col, xlabel, output_pdf, output_png):
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 9

    fig, axes = plt.subplots(1, len(ROUND_ORDER), figsize=(8.0, 2.25), sharey=True)
    for ax, round_name in zip(axes, ROUND_ORDER):
        obs = observed.loc[observed["first_observed_round"] == round_name, value_col].dropna()
        nul = null.loc[null["first_observed_round"] == round_name, value_col].dropna()
        all_values = pd.concat([obs, nul])
        x = np.linspace(all_values.min() - all_values.std() * 0.18, all_values.max() + all_values.std() * 0.18, 300)

        for values, label, color, linestyle in [
            (nul, "Null", "#A0A0A0", "--"),
            (obs, "Observed", ROUND_COLORS[round_name], "-"),
        ]:
            kde = gaussian_kde(values)
            y = kde(x)
            ax.plot(x, y, color=color, linewidth=1.8, linestyle=linestyle, label=label)
            ax.fill_between(x, y, color=color, alpha=0.12, linewidth=0)

        if value_col == "delta_ppl_star_vs_wt":
            ax.axvline(0, color="#777777", linewidth=0.7)
        ax.set_title(f"{round_name}\nobs n={len(obs)}", fontsize=9)
        ax.set_xlabel(xlabel if ax is axes[len(axes) // 2] else "")
        ax.grid(True, axis="y", color="#E7E7E7", linewidth=1)
        ax.tick_params(axis="both", length=0, labelsize=7.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)

    axes[0].set_ylabel("Density")
    axes[-1].legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_pdf)
    fig.savefig(output_png, dpi=300)
    plt.close(fig)


def main():
    observed = pd.read_csv(OBSERVED_CSV)
    rows = build_round_matched_null()
    null = score_rows(rows)
    wt_ppl = float(observed.loc[observed["variant_id"] == "WT", "single_pass_ppl_star"].iloc[0])
    null["delta_ppl_star_vs_wt"] = null["single_pass_ppl_star"] - wt_ppl

    null.to_csv(OUTPUT_NULL_CSV, index=False)
    summary = summarize(observed, null)
    summary.to_csv(OUTPUT_SUMMARY_CSV, index=False)
    plot_overlay(observed, null, "single_pass_ppl_star", "ESM2 PPL*", OUTPUT_PDF, OUTPUT_PNG)
    plot_overlay(observed, null, "delta_ppl_star_vs_wt", "Delta ESM2 PPL* vs WT", OUTPUT_DELTA_PDF, OUTPUT_DELTA_PNG)

    print(f"Wrote {OUTPUT_NULL_CSV}")
    print(f"Wrote {OUTPUT_SUMMARY_CSV}")
    print(f"Wrote {OUTPUT_PDF}")
    print(f"Wrote {OUTPUT_PNG}")
    print(f"Wrote {OUTPUT_DELTA_PDF}")
    print(f"Wrote {OUTPUT_DELTA_PNG}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
