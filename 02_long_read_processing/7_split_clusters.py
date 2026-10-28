import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import pipeline_settings as s  # noqa: E402


def clusters_from_tsv(tsv_path):
    # MMseqs2's createtsv output lists one representative-member pair per line, with all
    # members of a cluster consecutive. Same grouping as UMICseq-fast.py.
    current, members = None, []
    with open(tsv_path) as fh:
        for line in fh:
            if not line.strip():
                continue
            rep, member = line.rstrip("\n").split("\t")[:2]
            if rep != current:
                if current is not None:
                    yield members
                current, members = rep, []
            members.append(member)
    if current is not None:
        yield members


def main():
    p = s.parser("Write one FASTA per UMI cluster, for clusters inside the size window")
    p.add_argument("tsv", help="clusters_r<N>.tsv from 6_cluster_umis.py")
    p.add_argument("reads", help="that round's demultiplexed read FASTA")
    p.add_argument("--round", type=int, required=True, choices=s.ROUNDS)
    p.add_argument("--size-min", type=int, default=s.CLUSTER_SIZE_MIN)
    p.add_argument("--size-max", type=int, default=s.CLUSTER_SIZE_MAX)
    args = p.parse_args()

    out_dir = s.stage_dir(f"7_clusterfiles/round_{args.round}/clusters_clusterfiles")
    if not s.require(args.tsv, args.reads):
        return 1

    sizes = [len(c) for c in clusters_from_tsv(args.tsv)]
    kept = sum(1 for n in sizes if args.size_min <= n <= args.size_max)
    print(f"  {len(sizes)} clusters in {Path(args.tsv).name}; "
          f"{kept} with {args.size_min} <= members <= {args.size_max}")
    print(f"  -> {out_dir}/cluster_<i>.fasta")
    if args.dry_run:
        return s.done("7_split_clusters", True)

    reads = dict(s.read_fasta(args.reads))
    count, written = 0, 0
    for members in clusters_from_tsv(args.tsv):
        if not (args.size_min <= len(members) <= args.size_max):
            continue
        records = [(m, reads[m]) for m in members if m in reads]
        s.write_fasta(out_dir / f"cluster_{count}.fasta", records)
        written += len(records)
        count += 1
    print(f"  {count} cluster files, {written} reads placed")
    return s.done(f"7_split_clusters (round {args.round})", count == kept)


if __name__ == "__main__":
    raise SystemExit(main())
