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
    parser = argparse.ArgumentParser(description="Run DiMSum once per droplet gate to produce per-variant fitness scores")
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
