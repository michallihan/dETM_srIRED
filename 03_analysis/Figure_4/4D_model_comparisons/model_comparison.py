"""
Shared pieces of the model comparison: the variant set, kcat(app), and the MSA.

The scripts in this folder all speak about the same variants and score them the same way, so
the filtering rule, the fitness-to-kcat conversion and the mutation parsing live here rather
than in each script. Import it the way the other local helper modules in this repo are
imported, after the repo root is on sys.path.

Nothing here is run directly.
"""

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from repo_paths import KCAT_APP_FIT_JSON

THIS_DIR = Path(__file__).resolve().parent

# The alignment this folder works from: the top 5000 BLAST hits, aligned, with the srIRED
# wild type as the first record. 4A_MSA_trajectories holds the unaligned hits it was built
# from. EVmutation, GEMME and the MSA conservation baseline all read this file.
MSA_FASTA = THIS_DIR / "asr_blast_top5000_aligned.fasta"
MSA_FOCUS_SEQ = "seq00001"

# Written by 01_; every scoring script reads it.
VARIANT_TABLE_CSV = THIS_DIR / "variant_table_with_kcat_app.csv"

# One table per method, written by 02_ through 10_ and read by 11_.
MODEL_SCORES_DIR = THIS_DIR / "model_scores"
ESM2_SCORES_CSV = MODEL_SCORES_DIR / "esm2_single_pass_ppl.csv"
ESMC_SCORES_CSV = MODEL_SCORES_DIR / "esmc_single_pass_ppl.csv"
PROTEINMPNN_SCORES_CSV = MODEL_SCORES_DIR / "proteinmpnn_scores.csv"
EVMUTATION_SCORES_CSV = MODEL_SCORES_DIR / "evmutation_scores.csv"
THERMOMPNN_SCORES_CSV = MODEL_SCORES_DIR / "thermompnn_scores.csv"
MULTIEVOLVE_SCORES_CSV = MODEL_SCORES_DIR / "multievolve_fcn_scores.csv"
EVOLVEPRO_ONEHOT_SCORES_CSV = MODEL_SCORES_DIR / "evolvepro_rf_onehot_scores.csv"
EVOLVEPRO_ESM2_SCORES_CSV = MODEL_SCORES_DIR / "evolvepro_rf_esm2_650m_scores.csv"
GEMME_SCORES_CSV = MODEL_SCORES_DIR / "gemme_scores.csv"

# Intermediates shared between scripts.
VARIANT_FASTA = THIS_DIR / "variant_sequences.fasta"
ESM2_EMBEDDINGS_CSV = THIS_DIR / "esm2_t33_650M_mean_embeddings.csv"
THERMOMPNN_INFERENCE_CSV = THIS_DIR / "thermompnn_inference_5OCM.csv"
EVMUTATION_MODEL_DIR = THIS_DIR / "evmutation_model"

AA_ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")

# A low-confidence singleton driven by one selected replicate, absent from the
# high-confidence calls. Excluded from every step here.
EXCLUDED_VARIANTS = frozenset({"A271T"})


def kcat_app_fit():
    """The fitted fitness-to-kcat conversion, read from 01_fitness_score_processing."""
    with open(KCAT_APP_FIT_JSON) as handle:
        fit = json.load(handle)
    return float(fit["a"]), float(fit["b"])


def to_kcat_app(fitness_value):
    """Apparent kcat from selection fitness, using the repo's fitted conversion."""
    a, b = kcat_app_fit()
    return a * np.exp(b * fitness_value)


def kcat_app_error(kcat_app, sigma_final):
    """The fitness standard error propagated through that conversion."""
    _, b = kcat_app_fit()
    return abs(b) * kcat_app * sigma_final


def mutation_tokens(mut_id):
    """The mutation tokens of a variant name; wild type gives an empty list."""
    if not mut_id or mut_id == "WT" or str(mut_id).lower() == "nan":
        return []
    return str(mut_id).split("-")


def mutation_set(mut_id):
    return frozenset(mutation_tokens(mut_id))


def parse_mutation(token):
    """Parse "A123T" into ("A", 123, "T")."""
    if len(token) < 3:
        raise ValueError(f"Malformed mutation token: {token!r}")
    return token[0], int(token[1:-1]), token[-1]


def read_fasta(path):
    records = []
    name = None
    chunks = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(chunks)))
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if name is not None:
        records.append((name, "".join(chunks)))
    return records


def load_fitness_table(csv_path, require_sequence=True):
    """The low-confidence fitness table, filtered down to the variants scored here.

    Rows missing anything the comparison needs are dropped, a blank variant name becomes
    "WT", and the excluded singleton is removed. Applying this in one place is what keeps
    every method scoring the same variant set.
    """
    required = ["mut_id", "fitness_final", "sigma_final", "Nham_aa"]
    if require_sequence:
        required.append("aa_seq")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=required).copy()
    df["mut_id"] = df["mut_id"].astype(str).replace({"": "WT"})
    df = df[~df["mut_id"].isin(EXCLUDED_VARIANTS)].copy()
    if df["mut_id"].duplicated().any():
        duplicates = sorted(df.loc[df["mut_id"].duplicated(), "mut_id"])
        raise ValueError(f"Duplicate variant names in {Path(csv_path).name}: {duplicates[:5]}")
    if not df["mut_id"].eq("WT").any():
        raise ValueError(f"WT is absent from {Path(csv_path).name}")
    return df


def load_variant_table():
    """The table written by 01_, with a clear message if it has not been built yet."""
    if not VARIANT_TABLE_CSV.exists():
        raise SystemExit(f"Run 01_prepare_variant_table.py first: {VARIANT_TABLE_CSV.name} is missing.")
    return pd.read_csv(VARIANT_TABLE_CSV)


def wt_sequence(df):
    return str(df.loc[df["mut_id"] == "WT", "aa_seq"].iloc[0]).rstrip("*")


def wt_position_to_msa_column(wt_seq, aligned_wt):
    """Map 1-based wild-type residue positions onto 0-based alignment columns."""
    mapping = {}
    position = 0
    for col_idx, aa in enumerate(aligned_wt):
        if aa == "-":
            continue
        position += 1
        if position <= len(wt_seq):
            if aa != wt_seq[position - 1]:
                raise ValueError(
                    f"MSA reference mismatch at WT position {position}: "
                    f"dataset has {wt_seq[position - 1]!r}, MSA has {aa!r}"
                )
            mapping[position] = col_idx
    if len(mapping) < len(wt_seq):
        raise ValueError(f"Only mapped {len(mapping)} of {len(wt_seq)} WT residues into the MSA")
    return mapping


def msa_position_frequencies(wt_seq):
    """Pseudocounted per-position amino-acid frequencies over the alignment."""
    records = read_fasta(MSA_FASTA)
    if not records:
        raise ValueError(f"No records in {MSA_FASTA}")
    pos_to_col = wt_position_to_msa_column(wt_seq, records[0][1])
    pseudocount = 1.0
    denominator = len(records) + pseudocount * len(AA_ALPHABET)
    freqs = {}
    for pos, col_idx in pos_to_col.items():
        counts = Counter(seq[col_idx] for _, seq in records if seq[col_idx] in AA_ALPHABET)
        freqs[pos] = {aa: (counts.get(aa, 0) + pseudocount) / denominator for aa in AA_ALPHABET}
    return freqs


def variant_msa_score(mut_id, freqs):
    """Summed log frequency ratio of the mutant residue over the wild-type one."""
    score = 0.0
    for token in mutation_tokens(mut_id):
        from_aa, pos, to_aa = parse_mutation(token)
        site_freqs = freqs[pos]
        fallback = 1.0 / (len(freqs) * len(AA_ALPHABET))
        score += math.log(site_freqs.get(to_aa, fallback)) - math.log(site_freqs.get(from_aa, fallback))
    return score
