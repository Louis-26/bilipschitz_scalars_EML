# Code changes (stage 1: reproduction + bilipschitz embedder + noise)

Relative to branch `bilipschitz_embedded` @ 3a28122 (code under `ScalarEMLP_bilipschitz/`).

## `scalaremlp/nn/objax.py`: bilipschitz embedding rewritten
Before: `comp_inner_products[_jax](..., bilipschitz=True)` built $(XX^\top)^{1/2}$ sample by sample with
`torch.linalg.svd` of $XX^\top$ and concatenated a torch tensor with jax/numpy arrays.
- It failed even eagerly (`TypeError: concatenate requires ndarray ... got torch.Tensor`) and under `jax.grad`
  (`LinearizeTracer has no attribute __dlpack_device__`), so no HNN/N-ODE could train with it: the earlier
  "with embedding" Rockfish tables cannot come from this code (the env's egg-link pointed to another checkout,
  `/data/svillar3/ylu174/bilipschitz_scalars/ScalarEMLP`, which no longer exists).
- $XX^\top$ has rank ≤ 3, so $\sqrt{\cdot}$ of its zero eigenvalue has an infinite derivative.

After:
```python
def gram_sqrt_jax(x):                         # x: (n, 4, 3)
    U, S, _ = jnp.linalg.svd(x, full_matrices=False)
    return jnp.einsum('bik,bk,bjk->bij', U, S, U)   # (X X^T)^{1/2} = U S U^T
```
plus the numpy twin `gram_sqrt`; `comp_inner_products[_jax]`, `compute_scalars[_jax]` take
`bilipschitz=False` (default = upstream inner products), and `InvarianceLayer_objax(..., bilipschitz=False)`,
`EquivarianceLayer_objax(..., bilipschitz=False)` store and pass the flag. The old torch `compute_mat_sqrt`
is kept only for the notebooks.

### MLP activation option
`BasicMLP_objax(..., act="relu")` with `act in {relu, softplus, silu, tanh}`, passed through
`InvarianceLayer_objax(..., act=)` and `EquivarianceLayer_objax(..., act=)`; default ReLU = upstream.
Reason: the HNN vector field is $J\nabla H$, and a ReLU-MLP $H$ is piecewise linear in its inputs, so
$\nabla H$ is piecewise constant in feature space. With inner-product scalars the true $H$ is exactly linear
in the features (½|p|², ½(|q₁|-1)² = ½|q₁|² - |q₁| + ½, ...), so ReLU is harmless; with $(XX^\top)^{1/2}$ it
is quadratic and the ReLU bilipschitz HNN plateaus at rollout error ≈ 0.32 for every width/depth/lr.

## `experiments/trainer/hamiltonian_dynamics.py`
- Hamiltonian restored to upstream: `pe += k2*(|q1-q2|-l2)**2` (the branch had `.5*k2`, which changes the
  system and makes results incomparable with the paper; all older dataset caches were made with `.5*k2`).
- Dataset cache no longer depends on the working directory: `<ScalarEMLP_bilipschitz>/datasets/...`,
  overridable with `$SCALAREMLP_DATA_DIR`.

## New files
| file | purpose |
|---|---|
| `experiments/run_dsp.py` (options `--act`, `--rbf_gamma_scale` for the N-ODE RBF width) | CLI for one run: model, embedding, hyperparameters, seed, training noise; writes result.json + model.npz; computes val rollout (selection) and rollouts from perturbed test ICs |
| `experiments/prepare_data.py` | one-time dataset generation |
| `experiments/make_tasks.py` | task lists of the stages paper / tune / probe / tune2 / tune3 / final / noise |
| `experiments/summarize.py` | tables and figure in `output/` |
| `SLURM_execution/SLURM_script/run_tasks.sh` | job-array runner, `PER_GPU` runs per GPU, resumable |
| `SLURM_execution/SLURM_script/gpu_check.sh` | dataset generation + 50-epoch timing |

## Unchanged on purpose
Upstream entry points `hnn_scalars.py`, `neuralode_scalars.py` (still `num_epochs = 3` "for test purpose"),
the old `parameter_tuning/` scripts and results, trainers, loss, metric, learning-rate schedules.

## SLURM defaults: l40s everywhere
`run_tasks.sh`, `gpu_check.sh`, `script_template.sh`, `script_template_gpu_simple.sh`,
`ScalarEMLP_bilipschitz/script.sh`, `experiments/parameter_tuning/parameter_tune_{hnn,node}_1.slurm`:
partition `l40s`, account `enalisn1_gpu` (the old `svillar3_gpu` is no longer valid for this user),
≤ 8 CPU and ≤ 36 GB per GPU (billing stays at the GPU weight). `script_template.sh` converted from CRLF and its
mutually exclusive alternatives (`--mem-per-cpu`, `--gpus`, `--gpus-per-node`, `--gpu-bind`, `--constraint`)
commented out as `##SBATCH`, so it can be submitted as is. All scripts pass `sbatch --test-only`.

## ODE solver step cap
`trainer/hamiltonian_dynamics.py`: `BHamiltonianFlow` / `BOdeFlow` pass `mxstep=2000` (per output interval) to
`odeint`. Two bilipschitz N-ODE runs (3-100-5e-3, seed 0) hung > 5 h in the step-0 test rollout of the
untrained model because the adaptive step kept shrinking. Trained models need tens of steps per interval, so
the cap does not change any finished result; the dataset generator (`HamiltonianFlow`) is unchanged.
