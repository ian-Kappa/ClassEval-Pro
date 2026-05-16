"""Single entrypoint: run the full benchmark construction pipeline.

  python -m benchmark_construction.run               # all stages 1..5
  python -m benchmark_construction.run --from 4      # resume at stage 4
  python -m benchmark_construction.run --only 4 5    # just these stages
  python -m benchmark_construction.run --to 3        # stop after stage 3

Each stage reads the previous stage's artifact, so resuming is safe as long
as the upstream artifact files exist (benchmark_construction/artifacts/).
"""

import argparse

from . import (stage1_mine, stage2_extract, stage3_merge,
               stage4_compose, stage5_generate)

STAGES = {
    1: ("mine", stage1_mine.run),
    2: ("extract", stage2_extract.run),
    3: ("merge", stage3_merge.run),
    4: ("compose", stage4_compose.run),
    5: ("generate", stage5_generate.run),
}


def main():
    ap = argparse.ArgumentParser(description="ClassEval-Pro construction")
    ap.add_argument("--from", dest="start", type=int, default=1)
    ap.add_argument("--to", dest="end", type=int, default=5)
    ap.add_argument("--only", type=int, nargs="+",
                    help="run exactly these stage numbers")
    args = ap.parse_args()

    selected = (sorted(args.only) if args.only
                else [s for s in STAGES if args.start <= s <= args.end])

    for s in selected:
        name, fn = STAGES[s]
        print(f"\n===== stage {s}: {name} =====")
        fn()
    print("\n[done] pipeline finished")


if __name__ == "__main__":
    main()
