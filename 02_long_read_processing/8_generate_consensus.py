import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import pipeline_settings as s  # noqa: E402

# One cluster's consensus, as the campaign ran it: minimap2 and racon for a first draft,
# then medaka against that draft, then the VCF applied back to it.
TEMPLATE = (
    "minimap2 -x ava-ont {wt} {clusters}/cluster_{i}.fasta > aln_cluster_{i}.paf\n"
    "racon {clusters}/cluster_{i}.fasta aln_cluster_{i}.paf {wt} > racon_cluster_{i}.fasta\n"
    "mini_align -i {clusters}/cluster_{i}.fasta -r racon_cluster_{i}.fasta"
    " -p cluster_{i}.minialn -m\n"
    "medaka consensus cluster_{i}.minialn.bam cluster_{i}.probs --model {model}\n"
    "medaka variant racon_cluster_{i}.fasta cluster_{i}.probs cluster_{i}.vcf\n"
    "python3 {vcf2fasta} -v cluster_{i}.vcf -r racon_cluster_{i}.fasta"
    " -o {out}/cluster_{i}_cons.fasta\n"
)


def main():
    p = s.parser("Write the per-cluster consensus commands (auto.sh), and optionally run them")
    p.add_argument("--round", type=int, required=True, choices=s.ROUNDS)
    p.add_argument("--run", action="store_true",
                   help="run the generated script with sh; otherwise only write it")
    args = p.parse_args()

    work = s.stage_dir(f"7_clusterfiles/round_{args.round}")
    clusters = work / "clusters_clusterfiles"
    out = s.stage_dir(f"8_consensus/round_{args.round}/test")
    script = work / "auto.sh"
    wt = s.WT_FASTA[args.round]

    if not s.require(clusters, wt):
        return 1
    if not Path(s.VCF2FASTA).exists():
        print(f"  missing input: {s.VCF2FASTA}  (see README: third_party/)")
        return 1

    n = sum(1 for _ in clusters.glob("cluster_*.fasta"))
    body = "".join(
        TEMPLATE.format(i=i, wt=wt, clusters=clusters, out=out,
                        model=s.MEDAKA_MODEL, vcf2fasta=s.VCF2FASTA)
        for i in range(n)
    )
    print(f"  {n} clusters -> {script} ({n * 6} commands)")
    if not args.dry_run:
        script.write_text(body)

    if not args.run:
        print(f"  not run; `cd {work} && sh auto.sh` when medaka is on PATH")
        return s.done(f"8_generate_consensus (round {args.round})", n > 0)

    if not s.require_tools("minimap2", "racon", "mini_align", "medaka"):
        return 1
    code = s.run(["sh", "auto.sh"], dry_run=args.dry_run, cwd=work)
    return s.done(f"8_generate_consensus (round {args.round})", code == 0)


if __name__ == "__main__":
    raise SystemExit(main())
