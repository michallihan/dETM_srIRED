import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import pipeline_settings as s  # noqa: E402


def convert(fastq_path, fasta_path):
    n = 0
    with open(fastq_path) as fin, open(fasta_path, "w") as fout:
        while True:
            header = fin.readline()
            if not header:
                break
            seq = fin.readline().rstrip("\n")
            fin.readline()
            fin.readline()
            if not seq:
                break
            fout.write(f">{header[1:].split()[0]}\n{seq}\n")
            n += 1
    return n


def main():
    p = s.parser("Convert filtered FASTQ to FASTA, dropping the quality lines")
    p.add_argument("fastq", help="filtered FASTQ from 1_filter_reads.py")
    p.add_argument("-o", "--output", help="output FASTA (default: work/2_fasta/<stem>.fasta)")
    args = p.parse_args()

    out = Path(args.output) if args.output else \
        s.stage_dir("2_fasta") / (Path(args.fastq).stem + ".fasta")
    out.parent.mkdir(parents=True, exist_ok=True)

    if not s.require(args.fastq):
        return 1
    print(f"  {args.fastq} -> {out}")
    if args.dry_run:
        return s.done("2_fastq_to_fasta", True)
    n = convert(args.fastq, out)
    print(f"  {n} reads written")
    return s.done("2_fastq_to_fasta", n > 0)


if __name__ == "__main__":
    raise SystemExit(main())
