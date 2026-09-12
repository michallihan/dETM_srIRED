"""
Splits the merged reads of each sequencing sample into the five selection rounds.

Every round of the campaign carries its own 20 nt identifier in the amplicon, so one
sequencing sample holds reads from all five rounds mixed together. UMIC-seq_helper aligns
each read against the five identifiers in rcbc.fasta and writes it to the round it matches,
with the alignment score thresholds the original run used (22 22, against a 20 nt barcode).

Merging is not done here: the reads arrive already merged from 01_NGS. UMIC-seq_helper reads
uncompressed fastq, so each merged file is decompressed to a scratch copy, split, and the
per-round outputs compressed again into the fastq/ folder that 2_run_dimsum.py reads.

UMIC-seq.py and UMIC-seq_helper.py are vendored next to this script (Paul Jannis Zurek,
pjz26@cam.ac.uk, v1.1.1). The helper needs numpy, biopython, tqdm and scikit-bio, and it
imports StripedSmithWaterman from skbio.alignment, which current scikit-bio no longer
exposes — 0.6.2 is the newest version that works. The helper also prints the scores of the
first 250 reads unconditionally, so it raises IndexError on an input smaller than that;
harmless on real samples, worth knowing if you test it on a toy file.

Reads the helper cannot assign to any round go to <sample>_none.fastq and are counted but
not kept, since DiMSum only reads the round files. The count is the useful part: it says how
much of a sample carried no recognisable round identifier.

Outputs:
    fastq/{Input,Sort}-<n>_round<1-5>.fastq.gz
"""

import argparse
import gzip
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))

import dimsum_settings as settings  # noqa: E402

HELPER = THIS_DIR / "UMIC-seq_helper.py"
BARCODES = THIS_DIR / "rcbc.fasta"
THRESHOLDS = ("22", "22")
OUTPUT_DIR = THIS_DIR / "fastq"

# The ten sequencing samples, as the merging step names them.
SAMPLES = tuple(f"{kind}-{n}" for kind in ("Input", "Sort") for n in (1, 2, 3, 4, 6))
ROUNDS = (1, 2, 3, 4, 5)


def merged_read_file(sample):
    """The merged fastq for one sample, accepting either spelling NGmerge produced."""
    for name in (f"{sample}_Merged.fastq.gz", f"{sample}_Merge.fastq.gz",
                 f"{sample}_Merged.gz", f"{sample}_Merge.gz"):
        candidate = settings.MERGED_READS_DIR / name
        if candidate.exists():
            return candidate
    return None


def demultiplex(sample, source, dry_run):
    """Split one sample into its five rounds. Returns the round files written."""
    expected = [OUTPUT_DIR / f"{sample}_round{r}.fastq.gz" for r in ROUNDS]
    command = ["python", str(HELPER), "demultiplex",
               "--barcodes", str(BARCODES),
               "--input", f"<{sample} merged, uncompressed>",
               "--threshs", *THRESHOLDS,
               "--output", sample]
    print(f"  {' '.join(command)}")
    if dry_run:
        return []

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        plain = scratch / f"{sample}_Merged.fastq"
        with gzip.open(source, "rb") as compressed, open(plain, "wb") as handle:
            shutil.copyfileobj(compressed, handle)

        subprocess.check_call(
            ["python", str(HELPER), "demultiplex",
             "--barcodes", str(BARCODES),
             "--input", str(plain),
             "--threshs", *THRESHOLDS,
             "--output", sample],
            cwd=str(scratch),
        )

        written = []
        for target in expected:
            produced = scratch / target.name.replace(".fastq.gz", ".fastq")
            if not produced.exists():
                continue
            with open(produced, "rb") as handle, gzip.open(target, "wb") as compressed:
                shutil.copyfileobj(handle, compressed)
            written.append(target)

        unassigned = scratch / f"{sample}_none.fastq"
        if unassigned.exists():
            with open(unassigned) as handle:
                reads = sum(1 for _ in handle) // 4
            print(f"  {reads} reads matched no round identifier")
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="print the commands and check the inputs, without splitting")
    args = parser.parse_args()

    print(f"merged reads from {settings.MERGED_READS_DIR}")
    print(f"round files into  {OUTPUT_DIR}")
    if not HELPER.exists():
        print(f"missing {HELPER.name} next to this script")
        return 1

    missing, total = [], 0
    for sample in SAMPLES:
        source = merged_read_file(sample)
        print(f"\n[{sample}] {source.name if source else 'merged file not found'}")
        if source is None:
            missing.append(sample)
            continue
        total += len(demultiplex(sample, source, args.dry_run))

    if missing:
        print(f"\nno merged reads for: {', '.join(missing)}")
        return 1
    print(f"\n{total} round files written" if not args.dry_run
          else f"\n{len(SAMPLES)} samples ready to split")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
