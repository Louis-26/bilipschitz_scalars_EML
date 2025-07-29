#!/bin/bash
#SBATCH --partition=a100
#SBATCH --job-name=parameter_tune_node_1
#SBATCH --qos=qos_gpu
#SBATCH --account=svillar3_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=05:00:00
#SBATCH --output=parameter_tune_node_1.out
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=ylu174@jhu.edu

module load anaconda3/2024.02-1
conda activate scalar_mlp
python node_para_tune.py