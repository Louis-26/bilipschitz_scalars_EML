#!/bin/bash
# One-time environment setup (Rockfish, linux + NVIDIA GPU).
# Creates the conda env `scalar_mlp_gpu` from the exported spec; the repo itself is used via PYTHONPATH,
# so no `pip install -e .` is needed (an old egg-link in the env pointing elsewhere is harmless: PYTHONPATH wins).
set -e
REPO_ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
source /data/svillar3/ylu174/Anaconda3/etc/profile.d/conda.sh   # adapt to your conda install

if ! conda env list | grep -q "^scalar_mlp_gpu "; then
    conda create -n scalar_mlp_gpu python=3.13 -y
    conda activate scalar_mlp_gpu
    pip install -r "${REPO_ROOT}/scalar_mlp_env/requirements_gpu.txt"   # jax[cuda12] 0.7, objax 1.8, torch, olive-oil-ml
else
    conda activate scalar_mlp_gpu
fi

# sanity check (run on a GPU node; the login node has no GPU and falls back to CPU)
PYTHONPATH="${REPO_ROOT}/ScalarEMLP_bilipschitz:${REPO_ROOT}/ScalarEMLP_bilipschitz/experiments" \
python -c "import jax, objax, oil, scalaremlp; print(jax.__version__, jax.devices(), scalaremlp.__file__)"
