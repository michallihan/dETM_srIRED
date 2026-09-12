"""
One DiMSum run per selection round, counting every variant in that round's read pool.

Each round is counted on its own so that a variant's read count can be followed across the
campaign. The reads are the round files 1_demultiplex_reads.py wrote: already merged, so
these runs are single-end (--paired F), unlike the DMS runs in 02_.

The five experimental designs differ only in which round's files they name; the replicate
structure is the same throughout, five experiment replicates from Input/Sort 1, 2, 3, 4
and 6.

DiMSum's fitness stage runs but nothing downstream uses its fitness output — what this
pipeline hands on is the per-variant count table. The merge_variants.xlsx in each round
folder is that table saved as Excel, which is the form the counts-per-round analysis reads.

DiMSum must be installed separately and on PATH.

Outputs, one folder per round:
    round<1-5>/variant_data_merge.tsv, report.html, merge_variants.xlsx
"""

import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))

import dimsum_settings as settings  # noqa: E402

ROUNDS = (1, 2, 3, 4, 5)
READS_DIR = THIS_DIR / "fastq"


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="print each command and check its inputs, without running DiMSum")
    parser.add_argument("--only", metavar="N", type=int, choices=ROUNDS,
                        help="run just one round")
    args = parser.parse_args()

    rounds = [r for r in ROUNDS if args.only in (None, r)]
    print(f"round reads from {READS_DIR}")

    failures = 0
    for round_number in rounds:
        code = settings.run(
            label=f"round{round_number}",
            fastq_dir=READS_DIR,
            design_path=THIS_DIR / f"experimental_design_r{round_number}.txt",
            output_dir=THIS_DIR / f"round{round_number}",
            paired=False,
            dry_run=args.dry_run,
        )
        failures += bool(code)

    print(f"\n{len(rounds) - failures} of {len(rounds)} rounds "
          f"{'checked' if args.dry_run else 'completed'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
