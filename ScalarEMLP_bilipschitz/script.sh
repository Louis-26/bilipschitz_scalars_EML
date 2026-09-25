#!/bin/bash
#SBATCH --job-name=MyJob
#SBATCH --time=24:0:0
#SBATCH --partition=l40s
#SBATCH --account=enalisn1_gpu
#SBATCH --qos=qos_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
# number of tasks (processes) per node
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=end
#SBATCH --mail-user=ylu174@jh.edu

#### load and unload modules you may need
# module unload openmpi/intel
# module load mvapich2/gcc/64/2.0b
module load anaconda
conda activate scalar_mlp
cd ScalarEMLP
python experiments/hnn_scalars.py

#### execute code and write output file to OUT-24log.
# time mpiexec ./code-mvapich.x > OUT-24log
echo "Finished with job $SLURM_JOBID"

#### mpiexec by default launches number of tasks requested