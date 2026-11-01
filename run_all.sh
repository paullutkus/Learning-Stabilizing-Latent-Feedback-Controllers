#!/usr/bin/env bash
# Every figure and table of the paper.
#
#   bash run_all.sh plot [profile]            re-draw the figures and tables from the cached results (about a minute)
#   bash run_all.sh run [profile] [jobs]      re-run every experiment, then re-draw
#
# profile: paper (default; the runs of the paper, ~150 CPU-hours, cached in results/paper/) or fast (reduced
# sweeps, about 2 hours on 8 cores; see README.md).  Outputs: figures/<profile>/*.pdf and tables/<profile>/*.tex.
# Each experiment can also be run (or re-drawn, --stage plot) on its own:
#   python -m experiments.<name> [--profile paper|fast] [--stage run|plot|all] [--jobs N] [--cpus SPEC]
set -euo pipefail
cd "$(dirname "$0")"
case ${1:-plot} in
    plot) STAGE=plot ;;
    run)  STAGE=all ;;
    *)    echo "usage: bash run_all.sh [plot [profile] | run [profile] [jobs]]" >&2; exit 1 ;;
esac
PROFILE=${2:-paper}
JOBS=${3:-8}
python -m experiments.motivating_example --profile "$PROFILE" --stage "$STAGE" --jobs 2          # Fig. 1
python -m experiments.nonnormal          --profile "$PROFILE" --stage "$STAGE"                    # Figs. 2, 3
python -m experiments.linear_models      --profile "$PROFILE" --stage "$STAGE" --jobs "$JOBS"     # Fig. 5, Tables 1, 4
python -m experiments.nonlinear_models   --profile "$PROFILE" --stage "$STAGE" --jobs "$JOBS"     # Fig. 6, Tables 5, 6
python -m experiments.multi_unstable     --profile "$PROFILE" --stage "$STAGE" --jobs "$JOBS"     # Fig. 7, Table 2
