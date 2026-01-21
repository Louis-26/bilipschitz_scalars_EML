#!/bin/bash
#SBATCH --job-name=template_execution      # Name of the job
#SBATCH --account=svillar3_gpu             # Account to charge resources to
#SBATCH --partition=a100                   # Partition (queue) to run in
#SBATCH --qos=qos_gpu                      # Quality of Service (priority class)

#SBATCH --nodes=1                          # Number of nodes
#SBATCH --ntasks=1                         # Number of MPI tasks
#SBATCH --ntasks-per-node=1                # Tasks per node
#SBATCH --cpus-per-task=6                  # CPU cores per task

#SBATCH --mem=48G                          # Total memory
#SBATCH --mem-per-cpu=8G                   # Memory per CPU core (alternative)

#SBATCH --time=02:00:00                    # Walltime limit (HH:MM:SS)

#SBATCH --gres=gpu:1                       # Request 1 GPU (classic method)
#SBATCH --gpus=1                           # Request GPUs (new syntax)
#SBATCH --gpus-per-node=1                  # GPUs per node
#SBATCH --gpu-bind=single:1                # Bind task to one GPU
#SBATCH --gpu-freq=high                    # Request high GPU frequency

#SBATCH --constraint=gpu                   # Node hardware constraint

#SBATCH --output=../SLURM_output/%x_slurm_%j.out            # Stdout file (%x=job name, %j=job ID)
#SBATCH --error=../SLURM_output/%x_slurm_%j.err             # Stderr file

#SBATCH --mail-type=BEGIN,END,FAIL                  # Email notifications
#SBATCH --mail-user=ylu174@alumni.jhu.edu            # Email address
