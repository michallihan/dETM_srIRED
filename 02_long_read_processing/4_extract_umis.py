import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import pipeline_settings as s  # noqa: E402


def main():
    p = s.parser("Extract the UMI from each read with UMIC-seq UMIextract")
    p.add_argument("fasta", help="one round's demultiplexed FASTA")
    p.add_argument("-o", "--output", help="output FASTA of UMIs "
                                          "(default: work/4_umis/<stem>_umis.fasta)")
    args = p.parse_args()

    out = Path(args.output) if args.output else \
        s.stage_dir("4_umis") / (Path(args.fasta).stem + "_umis.fasta")
    out.parent.mkdir(parents=True, exist_ok=True)

    if not s.require(args.fasta, s.UMI_PROBE, s.UMIC_SEQ):
        return 1

    cmd = [sys.executable, s.UMIC_SEQ, "UMIextract",
           "-i", args.fasta,
           "-o", out,
           "--probe", s.UMI_PROBE,
           "--umi_loc", s.UMI_LOC,
           "--umi_len", s.UMI_LEN,
           "--min_probe_score", s.UMI_MIN_PROBE_SCORE]
    code = s.run(cmd, dry_run=args.dry_run)
    return s.done("4_extract_umis", code == 0)


if __name__ == "__main__":
    raise SystemExit(main())
