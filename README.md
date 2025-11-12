# bilipschitz_scalars_EML
test bilipschitz embedding influence brought to scalar methods of equivariant machine learning models

# configuration setup
git clone https://github.com/weichiyao/ScalarEMLP.git
conda create -n scalar_mlp python=3.13 -y
conda activate scalar_mlp
cd ScalarEMLP 
pip install -e .
pip install git+https://github.com/mfinzi/olive-oil-ml

# if want to use GPU, it must be on a linux machine with NVIDIA GPU to utilize jax cuda version,
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
or alternatively,
pip install --upgrade "jax[cuda12]"

# check both jax and cuda are working
python -c "import torch; print(torch.version.cuda)"
python -c "import jax; print(jax.devices())"



module list
module load cuda
module load cudnn

# run in rockfish
1. cd data_svillar3/ylu174_file/ScalarEMLP/experiments/parameter_tuning
2. ml anaconda3/2024.02-1
3. conda activate scalar_mlp_gpu
4. sbatch parameter_tune_hnn_1.slurm
5. sbatch parameter_tune_node_1.slurm

pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

if on windows: need to use WSL to enable cuda support for jax

## change executable permission
if on linux: chmod -R +x git_script


## reference projects
A Practical Method for Constructing Equivariant Multilayer Perceptrons for Arbitrary Matrix Groups:
https://github.com/mfinzi/equivariant-MLP

Olive-Oil-ML
https://github.com/mfinzi/olive-oil-ml

Scalar-based multi-layer perceptron models
https://github.com/weichiyao/ScalarEMLP