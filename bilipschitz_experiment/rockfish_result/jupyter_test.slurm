#!/bin/bash
#SBATCH --partition=a100
#SBATCH --job-name=parameter_tune_hnn_1
#SBATCH --qos=qos_gpu
#SBATCH --account=svillar3_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:01:00
#SBATCH --mem-per-cpu=8G
#SBATCH --output=parameter_tune_hnn_1.out
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=ylu174@jhu.edu

module load anaconda3/2024.02-1
conda activate scalar_mlp_gpu
jupyter nbconvert --execute --to html jupyter_rockfish_test.ipynb