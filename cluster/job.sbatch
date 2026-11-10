#!/bin/bash
# Generic job body: runs `python <script> <args>` in the project environment. Submit through cluster/submit.sh,
# which supplies account, partition, GPUs, memory, time and the log file.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
# --requeue: ckpt jobs are stopped and requeued (GPU jobs every 8-9 h, or whenever the owner needs the node).
# The job then starts again from the top, so scripts must resume from their own checkpoints / skip finished configs.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source cluster/config.env

export TORCH_HOME="$HYAK_DIR/cache/torch"      # pretrained weights; keeps them out of the 10 GB home
export MPLCONFIGDIR="$HYAK_DIR/cache/matplotlib"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "job $SLURM_JOB_ID on $(hostname), start $(date -Is), restarts so far: ${SLURM_RESTART_COUNT:-0}"
echo "command: python $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU in this job"

"$HYAK_DIR/env/bin/python" "$@"
echo "finished $(date -Is)"
