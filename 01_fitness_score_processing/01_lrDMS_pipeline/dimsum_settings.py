import os
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR
while not (REPO_ROOT / ".project-root").exists():
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

LRDMS_DIR = THIS_DIR

# Barcode-identity map linking each nucleotide barcode to its variant, from the nanopore
# long-read data. Copied from the original runs for now; it will be produced by the
# nanopore pipeline once that is in the repo.
BARCODE_IDENTITY_PATH = LRDMS_DIR / "VIF_all_rounds_upd.txt"

# Reads are not held in the repo. 01_NGS is where the download and merging steps put them;
# override either location with an environment variable to read them from elsewhere.
RAW_READS_DIR = Path(os.environ.get("LRDMS_RAW_READS_DIR", LRDMS_DIR / "01_NGS" / "raw_reads"))
MERGED_READS_DIR = Path(os.environ.get("LRDMS_MERGED_READS_DIR", LRDMS_DIR / "01_NGS" / "merged_reads"))

# IRED coding sequence, 879 nt including the stop codon.
WILDTYPE_SEQUENCE = (
    "ATGCGGGACACCGATGTCACTGTCCTCGGCCTGGGTCTGATGGGCCAGGCTTTGGCAGGCGCCTTTCTCAAAGATGGGCAT"
    "GCGACGACCGTTTGGAATCGTTCTGAAGGCAAAGCCGGTCAGCTGGCGGAACAGGGTGCGGTACTGGCCAGCAGCGCCCGG"
    "GATGCGGCCGAGGCGAGCCCACTGGTAGTTGTTTGCGTTTCAGACCACGCCGCGGTTCGCGCGGTGCTGGATCCTCTGGGT"
    "GATGTTCTGGCGGGGCGTGTTCTGGTGAACCTCACCTCTGGAACTAGCGAACAGGCCCGCGCCACGGCGGAGTGGGCCGCT"
    "GAACGCGGGATTACATACCTTGACGGAGCTATCATGGCAATCCCGCAGGTTGTCGGCACAGCAGATGCATTCTTACTGTAC"
    "TCTGGGCCGGAAGCCGCTTATGAGGCGCACGAACCGACCTTACGGAGCCTGGGGGCGGGTACCACTTATTTGGGCGCCGAC"
    "CACGGCCTGTCATCTTTGTATGACGTCGCGTTGCTCGGCATCATGTGGGGCACGCTGAATTCTTTCCTCCACGGGGCGGCA"
    "TTGTTGGGCACAGCCAAAGTGGAAGCGACGACCTTTGCACCGTTCGCGAACCGTTGGATTGAAGCGGTAACCGGGTTTGTA"
    "TCCGCGTATGCTGGTCAGGTGGACCAGGGGGCCTACCCGGCACTGGACGCGACTATCGATACCCACGTGGCTACGGTAGAT"
    "CATCTGATTCACGAGTCCGAGGCCGCTGGCGTGAACACGGAGCTCCCACGCCTCGTGCGCACGCTGGCCGACCGTGCCCTG"
    "GCGGGAGGCCAGGGCGGTCTTGGTTACGCCGCTATGATTGAGCAGTTTCGGTCTCCTAGCTAA"
)

# Constant regions flanking the amplicon, trimmed by cutadapt.
ADAPTER_5_FIRST = "ATTCGATGGTGTCCGGGATCTCGACGCTCTCCCTTAGATGC"
ADAPTER_5_SECOND = "GAGTCGTATTAATTTCCTAATGCAGGAGTCGCATATAGTGAGTCG"

# Options common to every run.
COMMON_OPTIONS = {
    "retainIntermediateFiles": "T",
    "cutadapt5First": ADAPTER_5_FIRST,
    "cutadapt5Second": ADAPTER_5_SECOND,
    "cutadaptMinLength": "20",
    "cutadaptErrorRate": "0.6",
    "vsearchMinQual": "10",
    "numCores": "8",
    "indels": "all",
    "maxSubstitutions": "100",
    "mixedSubstitutions": "T",
    "barcodeErrorRate": "0.1",
    "fitnessMinInputCountAll": "0",
    "startStage": "1",
}

PROJECT_NAME = "DiMSum_Project"


def build_command(fastq_dir, design_path, output_dir, paired):
    """The DiMSum command for one run, as an argument list."""
    options = {
        "fastqFileDir": str(fastq_dir),
        "experimentDesignPath": str(design_path),
        "wildtypeSequence": WILDTYPE_SEQUENCE,
        "barcodeIdentityPath": str(BARCODE_IDENTITY_PATH),
        "paired": "T" if paired else "F",
        "outputPath": str(output_dir),
        "projectName": PROJECT_NAME,
        **COMMON_OPTIONS,
    }
    command = ["DiMSum"]
    for key, value in options.items():
        command += [f"--{key}", value]
    return command


def check_inputs(fastq_dir, design_path):
    """Names anything the run needs that is not there, so a run fails before DiMSum starts."""
    missing = []
    if not Path(design_path).exists():
        missing.append(f"experimental design: {design_path}")
    if not BARCODE_IDENTITY_PATH.exists():
        missing.append(f"barcode identity map: {BARCODE_IDENTITY_PATH}")
    if not Path(fastq_dir).is_dir():
        missing.append(f"read folder: {fastq_dir}")
    elif not any(Path(fastq_dir).glob("*.fastq*")):
        missing.append(f"no .fastq files in {fastq_dir}")
    return missing


def run(label, fastq_dir, design_path, output_dir, paired, dry_run):
    """Print, and unless dry_run also execute, one DiMSum run. Returns its exit code."""
    command = build_command(fastq_dir, design_path, output_dir, paired)
    # Abbreviate only the wild-type sequence; paths stay readable.
    shown = " ".join(
        f"{arg[:20]}...[{len(arg)} nt]" if arg == WILDTYPE_SEQUENCE else arg
        for arg in command
    )
    print(f"\n[{label}]\n  {shown}")

    missing = check_inputs(fastq_dir, design_path)
    if missing:
        print("  missing inputs:")
        for item in missing:
            print(f"    - {item}")
        if not dry_run:
            print("  skipped")
            return 1
        return 0

    if dry_run:
        print("  inputs present; not run (dry run)")
        return 0

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    try:
        return subprocess.call(command, cwd=str(output_dir))
    except FileNotFoundError:
        print("  DiMSum was not found on PATH — install it and try again")
        return 127
