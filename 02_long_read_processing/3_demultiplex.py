import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import pipeline_settings as s  # noqa: E402


def main():
    p = s.parser("Split the reads into one FASTA per selection round, by round barcode")
    p.add_argument("fasta", help="converted FASTA from 2_fastq_to_fasta.py")
    p.add_argument("--prefix", default="reads_",
                   help="output prefix; UMIC-seq appends _<barcode name>.fasta (default: reads_)")
    args = p.parse_args()

    out_dir = s.stage_dir("3_demultiplexed")
    if not (s.require(args.fasta, s.BARCODES_FASTA, s.UMIC_SEQ_HELPER)):
        return 1

    cmd = [sys.executable, s.UMIC_SEQ_HELPER, "demultiplex",
           "-i", args.fasta,
           "-o", out_dir / args.prefix,
           "--barcodes", s.BARCODES_FASTA]
    code = s.run(cmd, dry_run=args.dry_run)
    return s.done("3_demultiplex", code == 0)


if __name__ == "__main__":
    raise SystemExit(main())
