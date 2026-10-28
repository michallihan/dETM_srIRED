import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import pipeline_settings as s  # noqa: E402


def main():
    p = s.parser("Filter basecalled reads on length and quality with NanoFilt")
    p.add_argument("reads", help="basecalled FASTQ (calls.fastq), or several merged with cat")
    p.add_argument("--dataset", choices=sorted(s.DATASETS), default="campaign",
                   help="which filter window to apply (default: campaign)")
    p.add_argument("--quality", type=int,
                   help="override the quality cut-off; required for round2_resequencing, "
                        "whose cut-off was not recorded")
    p.add_argument("-o", "--output", help="output FASTQ (default: work/1_filtered/<dataset>.fastq)")
    args = p.parse_args()

    cfg = dict(s.DATASETS[args.dataset])
    if args.quality is not None:
        cfg["min_quality"] = args.quality
    if cfg["min_quality"] is None:
        print(f"the quality cut-off for {args.dataset} was never recorded -- pass --quality")
        return 1

    out = Path(args.output) if args.output else s.stage_dir("1_filtered") / f"{args.dataset}.fastq"
    out.parent.mkdir(parents=True, exist_ok=True)

    if not (s.require(args.reads) and s.require_tools("NanoFilt")):
        return 1

    cmd = ["NanoFilt",
           "--length", cfg["min_length"],
           "--maxlength", cfg["max_length"],
           "--quality", cfg["min_quality"],
           args.reads]
    code = s.run(cmd, dry_run=args.dry_run, stdout_path=out)
    return s.done(f"1_filter_reads ({args.dataset})", code == 0)


if __name__ == "__main__":
    raise SystemExit(main())
