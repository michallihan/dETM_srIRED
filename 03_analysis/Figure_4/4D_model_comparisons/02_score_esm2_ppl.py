"""
How surprising ESM2 finds each variant in the network.

Scores every variant with a single forward pass of ESM2-650M and records PPL*, the
pseudo-perplexity of the sequence under the model. This is the same quantity 4E_ESM_baseline
measures on the high-confidence variants; here it is computed over the low-confidence network
so it can be asked the harder question in 11_, whether the model ranks the two ends of a
single-mutation step in the right order.

Needs torch and fair-esm, and downloads the ESM2-650M weights on first use. 01_ must have run.

Outputs, written to model_scores/ next to this script:
    esm2_single_pass_ppl.csv
"""

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR
while not (REPO_ROOT / ".project-root").exists():
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

import math

import esm
import pandas as pd
import torch

import model_comparison as mc  # noqa: E402

INPUT_CSV = mc.VARIANT_TABLE_CSV
OUTPUT_CSV = mc.ESM2_SCORES_CSV

MODEL_NAME = "esm2_t33_650M_UR50D"
BATCH_SIZE = 8


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(device: torch.device):
    model_loader = getattr(esm.pretrained, MODEL_NAME)
    model, alphabet = model_loader()
    model.eval()
    model.to(device)
    return model, alphabet


def score_batch(model, alphabet, device: torch.device, batch_rows: list[dict]) -> list[dict]:
    batch_converter = alphabet.get_batch_converter()
    batch = [(row["variant_id"], row["sequence"]) for row in batch_rows]
    _, _, tokens = batch_converter(batch)
    tokens = tokens.to(device)

    with torch.no_grad():
        logits = model(tokens, repr_layers=[], return_contacts=False)["logits"]
        log_probs = torch.log_softmax(logits, dim=-1)

    scored = []
    for idx, row in enumerate(batch_rows):
        seq_len = len(row["sequence"])
        residue_tokens = tokens[idx, 1 : seq_len + 1]
        residue_log_probs = log_probs[idx, 1 : seq_len + 1, :].gather(1, residue_tokens[:, None]).squeeze(1)
        mean_log_prob = float(residue_log_probs.mean().detach().cpu().item())
        scored.append(
            {
                **row,
                "esm2_model": MODEL_NAME,
                "sequence_length": seq_len,
                "single_pass_mean_log_probability": mean_log_prob,
                "single_pass_negative_mean_log_probability": -mean_log_prob,
                "single_pass_ppl_star": math.exp(-mean_log_prob),
            }
        )
    return scored


def main() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    input_df = pd.read_csv(INPUT_CSV)
    rows = [
        {
            "variant_id": row["mut_id"],
            "sequence": str(row["aa_seq"]).rstrip("*"),
            "fitness_final": row["fitness_final"],
            "sigma_final": row["sigma_final"],
            "kcat_app": row["kcat_app"],
            "kcat_app_error": row["kcat_app_error"],
            "Nham_aa": row["Nham_aa"],
        }
        for _, row in input_df.iterrows()
    ]

    device = choose_device()
    print(f"Using device: {device}")
    model, alphabet = load_model(device)

    scored: list[dict] = []
    for start in range(0, len(rows), BATCH_SIZE):
        scored.extend(score_batch(model, alphabet, device, rows[start : start + BATCH_SIZE]))
        print(f"Scored {min(start + BATCH_SIZE, len(rows))}/{len(rows)}", flush=True)

    out = pd.DataFrame(scored)
    wt = out.loc[out["variant_id"].eq("WT")].iloc[0]
    out["ppl_star_over_wt"] = out["single_pass_ppl_star"] / float(wt["single_pass_ppl_star"])
    out["delta_ppl_star_vs_wt"] = out["single_pass_ppl_star"] - float(wt["single_pass_ppl_star"])
    out["fitness_delta_vs_wt"] = out["fitness_final"] - float(wt["fitness_final"])
    out["kcat_app_over_wt"] = out["kcat_app"] / float(wt["kcat_app"])
    out["delta_kcat_app_vs_wt"] = out["kcat_app"] - float(wt["kcat_app"])
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
