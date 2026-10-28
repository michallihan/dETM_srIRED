import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import pipeline_settings as s  # noqa: E402


def main():
    p = s.parser("Call each cluster's mutations against the reference, per round")
    p.add_argument("--round", type=int, required=True, choices=s.ROUNDS)
    p.add_argument("--clusterfolder",
                   help="folder of consensus FASTAs (default: work/8_consensus/round_<N>/test)")
    args = p.parse_args()

    r = args.round
    folder = Path(args.clusterfolder) if args.clusterfolder else \
        s.WORK_DIR / f"8_consensus/round_{r}/test"

    inputs = [folder, s.REFERENCE_FASTA, s.PROBE_FRAME_UP, s.PROBE_FRAME_DOWN,
              s.PROBE_UMI_UP[r], s.PROBE_UMI_DOWN[r]]
    if not s.require(*inputs):
        return 1
    if not Path(s.UMI_MUT_PATT_MATCH).exists():
        print(f"  missing input: {s.UMI_MUT_PATT_MATCH}  (see README: third_party/)")
        return 1

    # The script names its outputs after the cluster folder, so it is run from the folder
    # it should write into, with the consensus folder passed relative to that.
    out_dir = s.stage_dir(f"9_mutations/round_{r}")
    cmd = [sys.executable, s.UMI_MUT_PATT_MATCH,
           "-r", s.REFERENCE_FASTA,
           "--probe_UMI_down", s.PROBE_UMI_DOWN[r],
           "--probe_UMI_up", s.PROBE_UMI_UP[r],
           "--clusterfolder", folder,
           "--probe_frame_down", s.PROBE_FRAME_DOWN,
           "--probe_frame_up", s.PROBE_FRAME_UP]
    code = s.run(cmd, dry_run=args.dry_run, cwd=out_dir)
    print(f"  expected output: {out_dir / 'full_table_barcodes_mutations.csv'}")
    return s.done(f"9_call_mutations (round {r})", code == 0)


if __name__ == "__main__":
    raise SystemExit(main())
