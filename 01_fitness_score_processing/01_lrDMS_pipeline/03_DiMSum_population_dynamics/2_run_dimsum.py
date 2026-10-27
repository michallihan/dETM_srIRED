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
