#!/bin/bash
#SBATCH --partition=a100
#SBATCH --job-name=parameter_tune_hnn_1
#SBATCH --qos=qos_gpu
#SBATCH --account=svillar3_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=05:00:00
#SBATCH --output=parameter_tune_hnn_1.out
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=ylu174@jhu.edu

module load anaconda3/2024.02-1
module load cuda
module load cudnn/9.8.0
conda activate scalar_mlp
conda install -y -c nvidia cudnn=9.8.0
cd ../..
pip install -e .
pip install git+https://github.com/mfinzi/olive-oil-ml
cd experiments/parameter_tuning
python hnn_para_tune.py