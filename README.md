# bilipschitz_scalars_EML
test bilipschitz embedding influence brought to scalar methods of equivariant machine learning models

# configuration setup
1. git clone https://github.com/weichiyao/ScalarEMLP.git 

2. cd ScalarEMLP 

3. conda create -n scalar_mlp python=3.13

4. conda activate scalar_mlp

5. pip install -e .

6. pip install git+https://github.com/mfinzi/olive-oil-ml

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