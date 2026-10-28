import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import pipeline_settings as s  # noqa: E402


def main():
    p = s.parser("Join round 2's initial reads with its October re-sequencing")
    p.add_argument("initial", help="round 2 FASTA from the campaign sequencing")
    p.add_argument("resequencing", help="round 2 FASTA from the re-sequencing run")
    p.add_argument("-o", "--output",
                   help="joint FASTA (default: work/3_demultiplexed/reads__round2_joint.fasta)")
    args = p.parse_args()

    out = Path(args.output) if args.output else \
        s.stage_dir("3_demultiplexed") / "reads__round2_joint.fasta"
    out.parent.mkdir(parents=True, exist_ok=True)

    if not s.require(args.initial, args.resequencing):
        return 1
    print(f"  {args.initial} + {args.resequencing} -> {out}")
    if args.dry_run:
        return s.done("5_join_round2_reads", True)

    counts = []
    with open(out, "w") as fout:
        for src in (args.initial, args.resequencing):
            n = 0
            for name, seq in s.read_fasta(src):
                fout.write(f">{name}\n{seq}\n")
                n += 1
            counts.append(n)
    print(f"  {counts[0]} + {counts[1]} = {sum(counts)} reads")
    return s.done("5_join_round2_reads", sum(counts) > 0)


if __name__ == "__main__":
    raise SystemExit(main())
