#!/bin/bash
# One-time dataset generation: 5000 double-spring-pendulum trajectories integrated with the upstream Hamiltonian
# (seed 2021, exactly as upstream makeTrainerScalars). Nothing is downloaded: the data are simulated.
# Output: ScalarEMLP_bilipschitz/datasets/ODEDynamics/DoubleSpringPendulum/trajectories_5000_5_0.2_30.pt (~1.8 MB)
# Takes ~1 min on an A100, a few minutes on CPU.
set -e
REPO_ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
source /data/svillar3/ylu174/Anaconda3/etc/profile.d/conda.sh
conda activate scalar_mlp_gpu
export SCALAREMLP_DATA_DIR="${REPO_ROOT}/ScalarEMLP_bilipschitz/datasets"
cd "${REPO_ROOT}/ScalarEMLP_bilipschitz"
python experiments/prepare_data.py   # prints (5000, 5, 12) and the global std
