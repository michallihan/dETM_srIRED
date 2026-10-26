"""
The same question asked of ESM-C.

Scores every variant with a single forward pass of ESM-C and records PPL*, exactly as 02_ does
for ESM2. Two zero-shot protein language models of different lineage, asked the same thing, so
that 11_ can tell a property of one model from a property of the approach.

Prefers the EvolutionaryScale `esm` package and falls back to the Hugging Face checkpoint
through transformers. Scoring is resumable with --resume. 01_ must have run.

Outputs, written to model_scores/ next to this script:
    esmc_single_pass_ppl.csv
"""

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR
while not (REPO_ROOT / ".project-root").exists():
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

import argparse
import csv
import math

import pandas as pd
import torch

import model_comparison as mc  # noqa: E402

INPUT_CSV = mc.VARIANT_TABLE_CSV
OUTPUT_CSV = mc.ESMC_SCORES_CSV

DEFAULT_MODEL_NAME = "esmc_300m"
DEFAULT_HF_MODEL_NAME = "biohub/ESMC-300M"
BATCH_SIZE = 4


def choose_device(preferred: str = "auto") -> torch.device:
    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_name: str, device: torch.device):
    try:
        from esm.models.esmc import ESMC
        from esm.sdk.api import LogitsConfig
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "ESM-C scoring requires the EvolutionaryScale ESM package with "
            "`esm.models.esmc`. Install/update that package before running this script."
        ) from exc

    model = ESMC.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return model, LogitsConfig


def load_transformers_model(model_name: str, device: torch.device, local_files_only: bool):
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(model_name, local_files_only=local_files_only, trust_remote_code=True)
    model.eval()
    model.to(device)
    return model, tokenizer


def tensor_to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if hasattr(value, "__dict__"):
        for key, item in vars(value).items():
            if isinstance(item, torch.Tensor):
                setattr(value, key, item.to(device))
    return value


def sequence_log_probs(client, logits_config_cls, device: torch.device, variant_id: str, sequence: str) -> dict:
    from esm.sdk.api import ESMProtein

    protein = ESMProtein(sequence=sequence)
    protein_tensor = tensor_to_device(client.encode(protein), device)
    logits_config = logits_config_cls(sequence=True, return_embeddings=False)

    with torch.no_grad():
        logits_output = client.logits(protein_tensor, logits_config)

    logits = logits_output.logits.sequence
    tokens = protein_tensor.sequence
    if logits.ndim == 2:
        logits = logits.unsqueeze(0)
    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(0)

    seq_len = len(sequence)
    residue_logits = logits[0, 1 : seq_len + 1, :]
    residue_tokens = tokens[0, 1 : seq_len + 1]
    log_probs = torch.log_softmax(residue_logits, dim=-1)
    residue_log_probs = log_probs.gather(1, residue_tokens[:, None]).squeeze(1)
    mean_log_prob = float(residue_log_probs.mean().detach().cpu().item())
    return {
        "variant_id": variant_id,
        "sequence": sequence,
        "esmc_model": getattr(client, "model_name", None),
        "sequence_length": seq_len,
        "single_pass_mean_log_probability": mean_log_prob,
        "single_pass_negative_mean_log_probability": -mean_log_prob,
        "single_pass_ppl_star": math.exp(-mean_log_prob),
    }


def transformers_sequence_log_probs(
    model,
    tokenizer,
    device: torch.device,
    variant_id: str,
    sequence: str,
) -> dict:
    inputs = tokenizer(
        sequence,
        return_tensors="pt",
        return_special_tokens_mask=True,
    )
    inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
    model_inputs = {key: value for key, value in inputs.items() if key != "special_tokens_mask"}

    with torch.no_grad():
        outputs = model(**model_inputs)

    logits = outputs.logits[0]
    token_ids = inputs["input_ids"][0]
    attention_mask = inputs.get("attention_mask", torch.ones_like(token_ids))[0].bool()
    special_mask = inputs["special_tokens_mask"][0].bool()
    residue_mask = attention_mask & ~special_mask
    residue_logits = logits[residue_mask]
    residue_tokens = token_ids[residue_mask]
    log_probs = torch.log_softmax(residue_logits, dim=-1)
    residue_log_probs = log_probs.gather(1, residue_tokens[:, None]).squeeze(1)
    mean_log_prob = float(residue_log_probs.mean().detach().cpu().item())
    return {
        "variant_id": variant_id,
        "sequence": sequence,
        "sequence_length": len(sequence),
        "single_pass_mean_log_probability": mean_log_prob,
        "single_pass_negative_mean_log_probability": -mean_log_prob,
        "single_pass_ppl_star": math.exp(-mean_log_prob),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=["auto", "esm-sdk", "transformers"],
        default="auto",
        help="ESM-C implementation to use.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="ESM-C model name passed to ESMC.from_pretrained.")
    parser.add_argument("--hf-model", default=DEFAULT_HF_MODEL_NAME, help="Hugging Face ESM-C model name.")
    parser.add_argument("--local-files-only", action="store_true", help="Do not download Hugging Face model files.")
    parser.add_argument("--device", default="auto", help="Device to use: auto, cpu, cuda, or mps.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Progress reporting chunk size.")
    parser.add_argument("--resume", action="store_true", help="Append missing variants to an existing output CSV.")
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N unscored variants.")
    args = parser.parse_args()

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
    if args.resume and OUTPUT_CSV.exists():
        existing = pd.read_csv(OUTPUT_CSV)
        scored_ids = set(existing["variant_id"].astype(str))
        rows = [row for row in rows if str(row["variant_id"]) not in scored_ids]
        print(f"Resuming from {OUTPUT_CSV}: {len(scored_ids)} already scored, {len(rows)} remaining")
    if args.limit is not None:
        rows = rows[: args.limit]

    device = choose_device(args.device)
    print(f"Using device: {device}")

    backend = args.backend
    scorer = None
    model_label = args.model
    if backend in {"auto", "esm-sdk"}:
        try:
            client, logits_config_cls = load_model(args.model, device)
            scorer = lambda row: sequence_log_probs(  # noqa: E731
                client,
                logits_config_cls,
                device,
                row["variant_id"],
                row["sequence"],
            )
            backend = "esm-sdk"
        except ModuleNotFoundError:
            if backend == "esm-sdk":
                raise
    if scorer is None:
        model, tokenizer = load_transformers_model(args.hf_model, device, args.local_files_only)
        scorer = lambda row: transformers_sequence_log_probs(  # noqa: E731
            model,
            tokenizer,
            device,
            row["variant_id"],
            row["sequence"],
        )
        backend = "transformers"
        model_label = args.hf_model
    print(f"Using backend: {backend}")

    scored: list[dict] = []
    output_fields = [
        "variant_id",
        "sequence",
        "fitness_final",
        "sigma_final",
        "kcat_app",
        "kcat_app_error",
        "Nham_aa",
        "esmc_model",
        "esmc_backend",
        "sequence_length",
        "single_pass_mean_log_probability",
        "single_pass_negative_mean_log_probability",
        "single_pass_ppl_star",
    ]
    # Only append when resuming; a plain re-run must truncate the previous file
    # rather than write a second header block into the middle of it.
    append = args.resume and OUTPUT_CSV.exists()
    handle = OUTPUT_CSV.open("a" if append else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=output_fields)
    if not append:
        writer.writeheader()
    for idx, row in enumerate(rows, start=1):
        scored_row = scorer(row)
        out_row = {**row, **scored_row, "esmc_model": model_label, "esmc_backend": backend}
        writer.writerow({field: out_row.get(field, "") for field in output_fields})
        handle.flush()
        scored.append(out_row)
        if idx % args.batch_size == 0 or idx == len(rows):
            print(f"Scored {idx}/{len(rows)}", flush=True)
    handle.close()

    out = pd.read_csv(OUTPUT_CSV)
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
