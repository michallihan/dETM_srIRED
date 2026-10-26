"""
EVmutation, the coevolution model, scoring each variant against the alignment.

Fits a pairwise undirected graphical model to the alignment and scores a variant by the change
in its Hamiltonian: how much less plausible the substitutions make the sequence given both the
conservation of each site and the couplings between sites. The first of the three MSA-based
methods here, and the one that sees pairs of positions rather than each position alone.

Needs evcouplings. The fitted parameters are reused if present; the binary they live in is too
large for the repo, so a first run refits them with plmc from the alignment, which needs the
`plmc` binary on PATH. 01_ must have run.

Outputs, written to model_scores/ next to this script:
    evmutation_scores.csv
    evmutation_model/srired_evmutation_params.bin   (refitted if absent; not tracked)
"""

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR
while not (REPO_ROOT / ".project-root").exists():
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from evcouplings.couplings.model import CouplingsModel
from evcouplings.couplings.tools import run_plmc

import model_comparison as mc  # noqa: E402

if not hasattr(np, "int"):
    np.int = int

MSA_FASTA = mc.MSA_FASTA
INPUT_CSV = mc.VARIANT_TABLE_CSV
OUTPUT_CSV = mc.EVMUTATION_SCORES_CSV
MODEL_DIR = mc.EVMUTATION_MODEL_DIR
COUPLINGS_FILE = MODEL_DIR / "srired_evmutation_couplings.txt"
PARAM_FILE = MODEL_DIR / "srired_evmutation_params.bin"
PLMC_BINARY = Path("plmc")

FOCUS_SEQ = mc.MSA_FOCUS_SEQ
ALPHABET = "-ACDEFGHIKLMNPQRSTVWY"


def mutation_tokens(mut_id: str) -> list[str]:
    if not mut_id or mut_id == "WT" or str(mut_id).lower() == "nan":
        return []
    return str(mut_id).split("-")


def parse_mutation(token: str) -> tuple[int, str, str]:
    if len(token) < 3:
        raise ValueError(f"Malformed mutation token: {token!r}")
    return int(token[1:-1]), token[0], token[-1]


def read_fasta_first(path: Path) -> tuple[str, str]:
    name = ""
    chunks: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name:
                    break
                name = line[1:].split()[0]
            else:
                chunks.append(line)
    return name, "".join(chunks)


def wt_position_to_model_position(wt_sequence: str) -> dict[int, int]:
    focus_name, aligned_focus = read_fasta_first(MSA_FASTA)
    if focus_name != FOCUS_SEQ:
        raise ValueError(f"Expected first MSA record {FOCUS_SEQ!r}, found {focus_name!r}")
    mapping: dict[int, int] = {}
    wt_pos = 0
    for col_idx, aa in enumerate(aligned_focus, start=1):
        if aa == "-":
            continue
        wt_pos += 1
        if wt_pos <= len(wt_sequence):
            expected = wt_sequence[wt_pos - 1]
            if aa != expected:
                raise ValueError(
                    f"WT/MSA mismatch at WT position {wt_pos}: dataset {expected!r}, MSA {aa!r}"
                )
            mapping[wt_pos] = col_idx
    return mapping


def fit_model() -> None:
    if PARAM_FILE.exists() and COUPLINGS_FILE.exists():
        print(f"Using existing EVmutation model: {PARAM_FILE}")
        return
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    result = run_plmc(
        alignment=str(MSA_FASTA),
        couplings_file=str(COUPLINGS_FILE),
        param_file=str(PARAM_FILE),
        focus_seq=FOCUS_SEQ,
        alphabet=ALPHABET,
        theta=0.8,
        ignore_gaps=True,
        iterations=100,
        lambda_h=0.01,
        lambda_J=16.2,
        cpu=8,
        binary=str(PLMC_BINARY),
    )
    print(result)


def score_variants() -> None:
    model = CouplingsModel(str(PARAM_FILE))
    df = pd.read_csv(INPUT_CSV)
    wt_sequence = str(df.loc[df["mut_id"].eq("WT"), "aa_seq"].iloc[0]).rstrip("*")
    pos_map = wt_position_to_model_position(wt_sequence)
    rows = []
    for row in df.itertuples(index=False):
        mut_id = str(row.mut_id)
        substitutions = []
        for token in mutation_tokens(mut_id):
            pos, from_aa, to_aa = parse_mutation(token)
            substitutions.append((pos_map[pos], from_aa, to_aa))
        delta_h, delta_couplings, delta_fields = model.delta_hamiltonian(substitutions)
        rows.append(
            {
                "variant_id": mut_id,
                "evmutation_delta_hamiltonian": float(delta_h),
                "evmutation_delta_couplings": float(delta_couplings),
                "evmutation_delta_fields": float(delta_fields),
                "evmutation_naturalness": float(delta_h),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {OUTPUT_CSV} ({len(out)} rows)")


def main() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fit_model()
    score_variants()


if __name__ == "__main__":
    main()
