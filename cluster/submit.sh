#!/bin/bash
# Submit one Python script as a Slurm job. Run on Hyak from the project directory
# (or from the laptop: cluster\hyak.ps1 submit ...).
#
#   bash cluster/submit.sh experiments/coco_detector_eval.py --model frcnn_r50v2 --stride 1
#   bash cluster/submit.sh TIME=1-00:00:00 GPUS=a40:1 experiments/train_x.py --epochs 200
#
# Leading KEY=VALUE words override cluster/config.env for this job:
#   ACCOUNT PARTITION GPUS (0 = CPU job, or type:count) CPUS MEM TIME NAME
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/config.env

ACCOUNT=$SLURM_ACCOUNT PARTITION=$SLURM_PARTITION GPUS=$SLURM_GPUS CPUS=$SLURM_CPUS MEM=$SLURM_MEM TIME=$SLURM_TIME NAME=
while [[ "${1:-}" =~ ^(ACCOUNT|PARTITION|GPUS|CPUS|MEM|TIME|NAME)=(.*)$ ]]; do
    declare "${BASH_REMATCH[1]}=${BASH_REMATCH[2]}"
    shift
done
[ $# -ge 1 ] || { echo "usage: bash cluster/submit.sh [KEY=VALUE ...] <script.py> [script args]"; exit 2; }
[ -f "$1" ] || { echo "no such script: $1"; exit 2; }
[ "$ACCOUNT" != CHANGEME ] || { echo "set SLURM_ACCOUNT in cluster/config.env (see hyakalloc)"; exit 2; }
NAME=${NAME:-$(basename "$1" .py)}

mkdir -p logs results
opts=(--account="$ACCOUNT" --partition="$PARTITION" --cpus-per-task="$CPUS" --mem="$MEM" --time="$TIME"
      --job-name="$NAME" --output="logs/%x_%j.out")
[ "$GPUS" = 0 ] || opts+=(--gpus="$GPUS")
sbatch "${opts[@]}" cluster/job.sbatch "$@"
