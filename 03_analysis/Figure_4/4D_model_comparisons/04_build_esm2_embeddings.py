"""
ESM2 mean embeddings, the features the supervised models in 06_ are trained on.

Runs the same ESM2-650M forward pass as 02_ but keeps the layer-33 representation averaged
over the sequence, giving one vector per variant. Written out once and reused, because the
embedding is the expensive part and the model fitted on top of it is not.

Needs torch and fair-esm, and downloads the ESM2-650M weights on first use. 01_ must have run.

Outputs, written next to this script:
    variant_sequences.fasta
    esm2_t33_650M_mean_embeddings.csv
    esm2_t33_650M_mean_embedding_pt/    one tensor per variant
"""

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR
while not (REPO_ROOT / ".project-root").exists():
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import torch
from esm import FastaBatchedDataset, pretrained

import model_comparison as mc  # noqa: E402

FASTA = mc.VARIANT_FASTA
OUTPUT_CSV = mc.ESM2_EMBEDDINGS_CSV
PT_DIR = THIS_DIR / "esm2_t33_650M_mean_embedding_pt"
MODEL_NAME = "esm2_t33_650M_UR50D"
LAYER = 33


def write_fasta(df: pd.DataFrame) -> None:
    with FASTA.open("w", encoding="utf-8") as handle:
        for row in df.itertuples(index=False):
            handle.write(f">{row.mut_id}\n{str(row.aa_seq).rstrip('*')}\n")


def main() -> None:
    df = mc.load_variant_table()
    write_fasta(df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Loading {MODEL_NAME} on {device}")
    model, alphabet = pretrained.load_model_and_alphabet(MODEL_NAME)
    model.eval()
    model = model.to(device)

    dataset = FastaBatchedDataset.from_file(FASTA)
    batches = dataset.get_batch_indices(toks_per_batch=4096, extra_toks_per_seq=1)
    loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(),
        batch_sampler=batches,
    )
    PT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[pd.DataFrame] = []
    with torch.no_grad():
        for batch_idx, (labels, seqs, toks) in enumerate(loader, start=1):
            print(f"Embedding batch {batch_idx}/{len(batches)} ({len(labels)} sequences)")
            toks = toks.to(device)
            out = model(toks, repr_layers=[LAYER], return_contacts=False)
            reps = out["representations"][LAYER].detach().cpu()
            for i, label in enumerate(labels):
                seq_len = len(seqs[i])
                mean_rep = reps[i, 1 : seq_len + 1].mean(0)
                torch.save(
                    {"label": label, "model": MODEL_NAME, "layer": LAYER, "mean_representation": mean_rep},
                    PT_DIR / f"{label}.pt",
                )
                rows.append(pd.DataFrame([mean_rep.numpy()], index=[label]))

    embeddings = pd.concat(rows)
    embeddings.index.name = "mut_id"
    embeddings.to_csv(OUTPUT_CSV)
    print(f"Wrote {OUTPUT_CSV} with shape {embeddings.shape}")
    print(f"Wrote per-variant tensors to {PT_DIR}")


if __name__ == "__main__":
    main()
