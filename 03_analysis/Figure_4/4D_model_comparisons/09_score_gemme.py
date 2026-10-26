"""
GEMME, scoring each variant from the evolutionary history behind the alignment.

Where EVmutation fits couplings, GEMME works from the conservation and the evolutionary
distance at which each substitution appears, combining them into a single score per variant.
Run through the official implementation, with the per-position conservation trace computed
from the same alignment rather than from a structure-derived JET run.

Clone the official GEMME repository into third_party/gemme_official at the repo root and set
GEMME_ENV to the prefix of the conda environment holding its R and Python dependencies.
01_ must have run.

Outputs, written to model_scores/ next to this script:
    gemme_scores.csv
    gemme_run/   GEMME's working directory, not tracked
"""

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR
while not (REPO_ROOT / ".project-root").exists():
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

import os
import re
import shutil
import subprocess

import numpy as np
import pandas as pd

import model_comparison as mc  # noqa: E402

MSA_FASTA = mc.MSA_FASTA
GEMME_SRC = REPO_ROOT / "third_party" / "gemme_official"
GEMME_ENV = Path(os.environ.get("GEMME_ENV", ""))
WORK_DIR = THIS_DIR / "gemme_run"
GEMME_COMPAT_DIR = WORK_DIR / "gemme_path_compat"
OUTPUT_CSV = mc.GEMME_SCORES_CSV

AA = list("ACDEFGHIKLMNPQRSTVWY")


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    name = ""
    chunks: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name:
                    records.append((name, "".join(chunks)))
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if name:
        records.append((name, "".join(chunks)))
    return records


def write_fasta(records: list[tuple[str, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for name, seq in records:
            handle.write(f">{name}\n")
            for idx in range(0, len(seq), 80):
                handle.write(seq[idx : idx + 80] + "\n")


def mutation_tokens(mut_id: str) -> list[str]:
    if not mut_id or mut_id == "WT" or str(mut_id).lower() == "nan":
        return []
    return str(mut_id).split("-")


def parse_selected_scores(path: Path, score_col: str) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            parts = raw.strip().split()
            if len(parts) >= 2:
                rows.append({"mut_id": parts[0].replace(":", "-"), score_col: float(parts[1])})
    return pd.DataFrame(rows)


def max_mutation_position(df: pd.DataFrame) -> int:
    max_pos = 0
    for mut_id in df["mut_id"].astype(str):
        for token in mutation_tokens(mut_id):
            match = re.fullmatch(r"([A-Z])(\d+)([A-Z])", token)
            if not match:
                raise ValueError(f"Malformed mutation token: {token!r}")
            max_pos = max(max_pos, int(match.group(2)))
    return max_pos


def make_gemme_path(dest: Path) -> Path:
    compat = GEMME_COMPAT_DIR
    compat.mkdir(parents=True, exist_ok=True)
    for filename in ["computePred.R", "pred.R"]:
        shutil.copy2(GEMME_SRC / filename, compat / filename)
    shutil.copy2(GEMME_SRC / "data" / "blosum62p.txt", compat / "blosum62p.txt")
    alpha_dest = compat / "alphabets"
    if alpha_dest.exists():
        shutil.rmtree(alpha_dest)
    shutil.copytree(GEMME_SRC / "prescott" / "alphabets", alpha_dest)
    return compat


def make_msa_inputs(df: pd.DataFrame) -> tuple[Path, Path, Path, str]:
    records = read_fasta(MSA_FASTA)
    if not records:
        raise ValueError(f"No FASTA records found in {MSA_FASTA}")
    wt_seq = str(df.loc[df["mut_id"].eq("WT"), "aa_seq"].iloc[0]).rstrip("*")
    focus_name, focus_aligned = records[0]
    focus_ungapped = focus_aligned.replace("-", "")
    max_pos = max_mutation_position(df)
    if not focus_ungapped.startswith(wt_seq) or max_pos > len(wt_seq):
        raise ValueError(
            f"First MSA record {focus_name!r} does not match dataset WT after removing gaps."
        )
    if len(focus_ungapped) != len(wt_seq):
        print(
            f"Note: MSA reference has {len(focus_ungapped)} ungapped residues; "
            f"dataset WT has {len(wt_seq)}. Dataset mutations stop at position {max_pos}."
        )

    keep_cols = [idx for idx, aa in enumerate(focus_aligned) if aa != "-"]
    trimmed_records = []
    for name, seq in records:
        if len(seq) != len(focus_aligned):
            raise ValueError(f"MSA record {name!r} has length {len(seq)}, expected {len(focus_aligned)}")
        trimmed_records.append((name, "".join(seq[idx] for idx in keep_cols)))

    ali_path = WORK_DIR / "IRED_low_confidence_query_columns.fasta"
    write_fasta([("IRED", trimmed_records[0][1]), *trimmed_records[1:]], ali_path)

    mut_path = WORK_DIR / "IRED_low_confidence_variants.mut"
    with mut_path.open("w", encoding="utf-8") as handle:
        for mut_id in df["mut_id"].astype(str):
            tokens = mutation_tokens(mut_id)
            if tokens:
                handle.write(":".join(tokens) + "\n")

    jet_path = WORK_DIR / "IRED_low_confidence_jet_from_msa.res"
    make_msa_conservation_jet(trimmed_records, jet_path)
    return ali_path, mut_path, jet_path, wt_seq


def make_msa_conservation_jet(records: list[tuple[str, str]], jet_path: Path) -> None:
    aligned = [seq.upper() for _, seq in records]
    focus = aligned[0]
    rows = []
    ungapped_pos = 0
    for col_idx, wt_aa in enumerate(focus):
        if wt_aa == "-":
            continue
        ungapped_pos += 1
        column = [seq[col_idx] for seq in aligned if col_idx < len(seq)]
        residues = [aa for aa in column if aa in AA]
        n = len(residues)
        if n == 0:
            trace = 0.0
            wt_freq = 0.0
        else:
            counts = np.array([residues.count(aa) for aa in AA], dtype=float)
            freqs = counts / counts.sum()
            entropy = -float(np.sum(freqs[freqs > 0] * np.log(freqs[freqs > 0])))
            trace = 1.0 - entropy / np.log(len(AA))
            wt_freq = residues.count(wt_aa) / n
        rows.append(
            {
                "residue": ungapped_pos,
                "chain": "A",
                "trace": round(trace, 4),
                "pc": round(trace, 4),
                "cv": 0.0,
                "clusters": 0.0,
                "freq": round(wt_freq, 4),
            }
        )
    pd.DataFrame(rows).to_csv(jet_path, sep="\t", index=False)


def run_gemme(ali_path: Path, mut_path: Path, jet_path: Path, compat_path: Path) -> None:
    env = os.environ.copy()
    tmp_dir = WORK_DIR / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    mpl_dir = WORK_DIR / "matplotlib"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    env["GEMME_PATH"] = str(compat_path)
    env["PATH"] = f"{GEMME_ENV / 'bin'}:{env.get('PATH', '')}"
    env["MPLCONFIGDIR"] = str(mpl_dir)
    env["TMPDIR"] = str(tmp_dir)
    env["TEMP"] = str(tmp_dir)
    env["TMP"] = str(tmp_dir)
    cmd = [
        str(GEMME_ENV / "bin" / "python"),
        str(GEMME_SRC / "gemme.py"),
        ali_path.name,
        "--mutations",
        mut_path.name,
        "--jetfile",
        jet_path.name,
        "--normweightmode",
        "trace",
        "--alphabet",
        "lw-i.7",
    ]
    subprocess.run(cmd, cwd=WORK_DIR, env=env, check=True)


def main() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    if not GEMME_SRC.is_dir():
        raise SystemExit(f"Clone the official GEMME repository into {GEMME_SRC} before running this step.")
    if not (GEMME_ENV / "bin" / "python").exists():
        raise SystemExit("Set GEMME_ENV to the GEMME conda environment prefix before running this step.")
    df = mc.load_variant_table()

    compat_path = make_gemme_path(WORK_DIR)
    ali_path, mut_path, jet_path, _ = make_msa_inputs(df)

    score_path = WORK_DIR / "IRED_normPred_evolCombi.txt"
    if not score_path.exists():
        run_gemme(ali_path, mut_path, jet_path, compat_path)
    scores = parse_selected_scores(score_path, "gemme_official_evolcombi_score")
    out = pd.DataFrame({"mut_id": df["mut_id"]})
    out = out.merge(scores, on="mut_id", how="left")
    out.loc[out["mut_id"].eq("WT"), "gemme_official_evolcombi_score"] = 0.0
    out["gemme_official_input"] = "official_gemme_core_with_msa_derived_jet_trace"
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {OUTPUT_CSV} ({len(out)} rows)")
    print(f"GEMME work directory: {WORK_DIR}")


if __name__ == "__main__":
    main()
