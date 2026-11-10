#!/bin/bash
# One-time (and after requirements-hyak.txt changes): build the Python environment in $HYAK_DIR/env.
# Run on a COMPUTE node (klone has no `module` command on login nodes), from the project directory:
#   salloc --account=<account> --partition=ckpt-all --cpus-per-task=4 --mem=16G --time=1:00:00
#   cd <HYAK_DIR> && bash cluster/setup_env.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/config.env

# keep conda/pip caches out of the 10 GB home directory
export CONDA_PKGS_DIRS="$HYAK_DIR/cache/conda"
export PIP_CACHE_DIR="$HYAK_DIR/cache/pip"
mkdir -p "$HYAK_DIR/cache"

if [ ! -x "$HYAK_DIR/env/bin/python" ]; then
    module load conda
    conda create -y --prefix "$HYAK_DIR/env" python=3.13 pip
fi
"$HYAK_DIR/env/bin/python" -m pip install -r cluster/requirements-hyak.txt

"$HYAK_DIR/env/bin/python" - <<'EOF'
import torch, torchvision
print('torch', torch.__version__, '| torchvision', torchvision.__version__, '| built with CUDA', torch.version.cuda)
print('GPU visible in this session:', torch.cuda.is_available())
EOF
echo "Environment ready: $HYAK_DIR/env"
