This is where we reproduce the results of the original work.
# bilipschitz_scalars_EML
test bilipschitz embedding influence brought to scalar methods of equivariant machine learning models

# main task
1. understand bilipschitz models(replace original inner product matrix by SVD square root)✅ 

2. tune bilipschitz model hyperparameters and compare performance✅

3. add some noise with controlled amplitude and compare regression results

4. add adversarial attack and compare regression results

# configuration setup
## Prerequisite
- NVIDIA GPU
- CUDA

in HPC, run this and configure the environment
```bash
srun -p brtx6 --gres=gpu:1 --time=02:00:00 --pty bash
```


```bash
git clone https://github.com/weichiyao/ScalarEMLP.git
conda create -n scalar_mlp python=3.13 -y
conda activate scalar_mlp
cd ScalarEMLP 
pip install -e .
pip install git+https://github.com/mfinzi/olive-oil-ml
```

if you want to use GPU, it must be on a linux machine with NVIDIA GPU to utilize jax cuda version,

if on windows: need to use WSL to enable cuda support for jax

```bash
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
or alternatively,
```bash
pip install --upgrade "jax[cuda12]"
```

# check both jax and cuda are working
```bash
python -c "import torch; print(torch.version.cuda)"
python -c "import jax; print(jax.devices())"
```


It is supposed to have pytorch version number(e.g., 12.8), 
and jax device([CudaDevice(id=0)])


# run in rockfish
1. cd data_svillar3/ylu174_file/ScalarEMLP/experiments/parameter_tuning
2. ml anaconda3/2024.02-1
3. conda activate scalar_mlp_gpu
4. sbatch parameter_tune_hnn_1.slurm
5. sbatch parameter_tune_node_1.slurm



## change executable permission
if on linux: 
```bash
chmod -R +x git_script
```


## reference projects
A Practical Method for Constructing Equivariant Multilayer Perceptrons for Arbitrary Matrix Groups:
https://github.com/mfinzi/equivariant-MLP

Olive-Oil-ML
https://github.com/mfinzi/olive-oil-ml

Scalar-based multi-layer perceptron models
https://github.com/weichiyao/ScalarEMLP

## project record
google slide: https://docs.google.com/presentation/d/1Ndx5qn9NyZEM6DsGkkfvgJvxYr85DGAVJ5ZrlwyTBa4/edit?usp=sharing


overleaf: 

original capstone report: https://www.overleaf.com/read/qbcjrjqgghts#f5b69b

current formal project paper draft: 