"""
DiMSum runs that turn the sorted droplet libraries into per-variant fitness scores.

The library was sorted through two FACS gates and sequenced as six sort samples against
five input samples. Sorts 1-3 are the lenient gate, sorts 4-6 the stringent gate. Reads go
into DiMSum as raw pairs; DiMSum merges them itself, so nothing upstream of this merges or
demultiplexes them.

Two runs:

    lenient_gate     three replicates, Sort-1/2/3 against Input-1/2/3
    stringent_gate   three replicates, Sort-4/5/6 against Input-4/2/6

These are the pair that 02a_ and 02b_ combine into fitness_final, reading each run's
all_variants.csv directly.

An earlier two-replicate run of the stringent gate (Sort-4/6 against Input-4/6) also exists
in the source folders. It is not here: every stringent value in both final tables comes from
the three-replicate run, and the earlier run holds no variant that run lacks.

The experimental designs here were reconstructed, because the design files the runs
actually used are not among the surviving files. The replicate structure is certain: it is
read off the count columns of each run's own output (input4_e1_s0, input2_e2_s0,
input6_e3_s0 and so on). The read filenames follow the MiSeq naming of the eight raw files
that do survive; three of them, Sort-1_S1, Sort-2_S2 and Sort-3_S3, are inferred from that
pattern and should be checked against whatever the deposited reads are named.

Reads are not in the repo. Put them in 01_NGS/raw_reads, or point LRDMS_RAW_READS_DIR
elsewhere. DiMSum must be installed separately and on PATH.

Outputs, one folder per run:
    lenient_gate/ stringent_gate/
        variant_data_merge.tsv, all_variants.csv, fitness_*.txt, report.html
"""

import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))

import dimsum_settings as settings  # noqa: E402

# (label, design file, output folder)
RUNS = (
    ("lenient_gate", "experimental_design_lenient_gate.txt", "lenient_gate"),
    ("stringent_gate", "experimental_design_stringent_gate.txt", "stringent_gate"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="print each command and check its inputs, without running DiMSum")
    parser.add_argument("--only", metavar="LABEL", choices=[run[0] for run in RUNS],
                        help="run just one of: " + ", ".join(run[0] for run in RUNS))
    args = parser.parse_args()

    runs = [run for run in RUNS if args.only in (None, run[0])]
    print(f"reads from {settings.RAW_READS_DIR}")

    failures = 0
    for label, design, output in runs:
        code = settings.run(
            label=label,
            fastq_dir=settings.RAW_READS_DIR,
            design_path=THIS_DIR / design,
            output_dir=THIS_DIR / output,
            paired=True,
            dry_run=args.dry_run,
        )
        failures += bool(code)

    print(f"\n{len(runs) - failures} of {len(runs)} runs "
          f"{'checked' if args.dry_run else 'completed'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
