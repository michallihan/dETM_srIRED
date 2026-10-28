# Shared settings and helpers for the nanopore pipeline in this folder.
#
# Parameters below are marked [recovered] where they were read back off the run's own
# outputs, [documented] where they come only from the lab handbooks, and [unknown] where
# neither source has them. See README.md in this folder.

import shutil
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR
while not (REPO_ROOT / ".project-root").exists():
    REPO_ROOT = REPO_ROOT.parent

INPUTS_DIR = THIS_DIR / "inputs"
WORK_DIR = THIS_DIR / "work"

ROUNDS = (1, 2, 3, 4, 5)

# The campaign's five rounds were sequenced together in June 2023 and basecalled in July.
# Round 2 was re-sequenced in October 2023 and the two read sets joined; the re-sequencing
# was filtered on a different length window, so it is its own dataset here.
DATASETS = {
    "campaign": {
        "min_length": 1500,     # [recovered] min read length in trimmed_reads.fastq
        "max_length": 3000,     # [recovered] max read length, and NanoPlot's top-5 longest
        "min_quality": 8,       # [recovered] min of NanoFilt's average read quality
    },
    "round2_resequencing": {
        "min_length": 2000,     # [recovered] min read length in reads__round2_resequences.fasta
        "max_length": 3500,     # [recovered] max read length in the same file
        "min_quality": None,    # [unknown] only FASTA survives for this run; pass --quality
    },
}

# UMI extraction, UMIC-seq.py UMIextract
UMI_PROBE = INPUTS_DIR / "probe.fasta"
UMI_LOC = "up"
UMI_LEN = 88                   # [recovered] UMIS_*.fasta wrap to 60 + 28 characters
UMI_MIN_PROBE_SCORE = 60       # [documented]

# MMseqs2 clustering
MMSEQS_MIN_SEQ_ID = 0.8        # [documented] no log survives; see README
MMSEQS_COVERAGE = 0.8          # [documented]
MMSEQS_COV_MODE = 0            # [documented]
MMSEQS_SEQ_ID_MODE = 2         # [documented]

# Cluster size window for the split into per-UMI fasta files
CLUSTER_SIZE_MIN = 20          # [recovered] only >=20 reproduces the cluster counts
CLUSTER_SIZE_MAX = 1_000_000   # [documented]

# Consensus generation
MEDAKA_MODEL = "r1041_e82_400bps_sup_v4.2.0"   # [recovered] from auto.sh
# wt1..wt5.fasta differ per round: each is the restriction-digested fragment for that round.
WT_FASTA = {r: INPUTS_DIR / f"wt{r}.fasta" for r in ROUNDS}

# Mutation calling, UMI_mut_patt_match_V6_Rike1.py
REFERENCE_FASTA = INPUTS_DIR / "reference.fasta"
PROBE_FRAME_UP = INPUTS_DIR / "probe_frame_up.fa"
PROBE_FRAME_DOWN = INPUTS_DIR / "probe_frame_down.fa"
PROBE_UMI_UP = {r: INPUTS_DIR / f"probe_UMI_up_r{r}.fa" for r in ROUNDS}
PROBE_UMI_DOWN = {1: INPUTS_DIR / "probe_UMI_down_r1.fa",
                  **{r: INPUTS_DIR / "probe_UMI_down_r2345.fa" for r in (2, 3, 4, 5)}}

BARCODES_FASTA = INPUTS_DIR / "barcodes.fasta"

# The mutation caller's own output per round, as it ran for the published dictionary:
# rounds 1 and 2 from the December 2023 re-analysis, rounds 3-5 from August 2023. These are
# what 10_ joins when work/9_mutations/ is empty, so 10_ and 11_ run from a clone.
PER_ROUND_DIR = THIS_DIR / "per_round"
PER_ROUND_TABLE = {r: PER_ROUND_DIR / f"round_{r}" / "full_table_barcodes_mutations.csv"
                   for r in ROUNDS}

# Third-party scripts. UMIC-seq is vendored in the repo already, under the lrDMS pipeline.
UMIC_SEQ = (REPO_ROOT / "01_fitness_score_processing" / "01_lrDMS_pipeline"
            / "03_DiMSum_population_dynamics" / "UMIC-seq.py")
UMIC_SEQ_HELPER = UMIC_SEQ.with_name("UMIC-seq_helper.py")
# Not in the repo; see README for where to put them.
THIRD_PARTY = REPO_ROOT / "third_party"
VCF2FASTA = THIRD_PARTY / "vcf2fasta.py"
UMI_MUT_PATT_MATCH = THIRD_PARTY / "UMI_mut_patt_match_V6_Rike1.py"


def stage_dir(name, create=True):
    d = WORK_DIR / name
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def require(*paths):
    missing = [p for p in paths if not Path(p).exists()]
    for p in missing:
        print(f"  missing input: {p}")
    return not missing


def require_tools(*tools):
    missing = [t for t in tools if shutil.which(t) is None]
    for t in missing:
        print(f"  not on PATH: {t}")
    return not missing


def run(cmd, dry_run=False, cwd=None, stdout_path=None):
    printable = " ".join(str(c) for c in cmd)
    if stdout_path:
        printable += f" > {stdout_path}"
    print(f"  {printable}")
    if dry_run:
        return 0
    if stdout_path:
        with open(stdout_path, "w") as fh:
            return subprocess.run([str(c) for c in cmd], cwd=cwd, stdout=fh).returncode
    return subprocess.run([str(c) for c in cmd], cwd=cwd).returncode


def parser(description):
    import argparse
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--dry-run", action="store_true",
                   help="print the commands and check inputs, without running anything")
    return p


def read_fasta(path):
    name, chunks = None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(chunks)
                name, chunks = line[1:], []
            else:
                chunks.append(line)
    if name is not None:
        yield name, "".join(chunks)


def write_fasta(path, records, wrap=60):
    with open(path, "w") as fh:
        for name, seq in records:
            fh.write(f">{name}\n")
            if wrap:
                for i in range(0, len(seq), wrap):
                    fh.write(seq[i:i + wrap] + "\n")
            else:
                fh.write(seq + "\n")


def done(label, ok):
    print(f"\n{label}: {'ok' if ok else 'FAILED'}")
    return 0 if ok else 1
