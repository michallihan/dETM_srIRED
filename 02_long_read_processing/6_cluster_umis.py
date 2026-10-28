import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import pipeline_settings as s  # noqa: E402


def main():
    p = s.parser("Cluster one round's UMIs with MMseqs2")
    p.add_argument("umis", help="FASTA of UMIs from 4_extract_umis.py")
    p.add_argument("--round", type=int, required=True, choices=s.ROUNDS,
                   help="which round these UMIs come from; names the outputs")
    args = p.parse_args()

    r = args.round
    out_dir = s.stage_dir(f"6_clusters/round_{r}")
    umi_db = out_dir / f"UMI_r{r}.DB"
    clus_db = out_dir / f"clus_r{r}.DB"
    tsv = out_dir / f"clusters_r{r}.tsv"

    if not (s.require(args.umis) and s.require_tools("mmseqs")):
        return 1

    steps = [
        ["mmseqs", "createdb", args.umis, umi_db],
        ["mmseqs", "cluster", umi_db, clus_db, out_dir / "tmp",
         "-c", s.MMSEQS_COVERAGE,
         "--cov-mode", s.MMSEQS_COV_MODE,
         "--min-seq-id", s.MMSEQS_MIN_SEQ_ID,
         "--seq-id-mode", s.MMSEQS_SEQ_ID_MODE],
        ["mmseqs", "createtsv", umi_db, umi_db, clus_db, tsv],
    ]
    for cmd in steps:
        code = s.run(cmd, dry_run=args.dry_run)
        if code:
            return s.done("6_cluster_umis", False)
    return s.done(f"6_cluster_umis (round {r})", True)


if __name__ == "__main__":
    raise SystemExit(main())
