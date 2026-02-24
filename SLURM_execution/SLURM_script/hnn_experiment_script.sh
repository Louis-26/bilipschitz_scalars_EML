#!/bin/bash
#SBATCH --job-name=hnn_experiment
#SBATCH --partition=brtx6
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=ylu174@alumni.jhu.edu
#SBATCH --output=../SLURM_output/%x_%j.out            # Stdout file (%x=job name, %j=job ID)
#SBATCH --error=../SLURM_output/%x_%j.err             # Stderr file

source /brtx/605-nvme2/ylu174/Anaconda3/etc/profile.d/conda.sh
conda activate scalar_mlp
python ../../bilipschitz_experiment/ScalarEMLP/experiments/hnn_scalars.py