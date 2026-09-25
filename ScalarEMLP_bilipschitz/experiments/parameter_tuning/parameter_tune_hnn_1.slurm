#!/bin/bash
#SBATCH --partition=l40s
#SBATCH --job-name=parameter_tune_hnn_1
#SBATCH --qos=qos_gpu
#SBATCH --account=enalisn1_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=05:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --output=parameter_tune_hnn_1.out
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=ylu174@jhu.edu

module load anaconda3/2024.02-1
conda activate scalar_mlp_gpu
python hnn_para_tune.py