# Bilipschitz scalars for equivariant dynamics learning: overview

**Question.** Scalar-based equivariant models (Villar et al. 2021; Yao et al. 2021) feed the pairwise inner
products $XX^\top$ of the input vectors to an MLP. The Gram map is invariant but not bilipschitz (quadratic).
Does replacing it by the bilipschitz invariant $\theta(X)=(XX^\top)^{1/2}$ (Balan & Dock 2022) help, in
particular under noisy data?

**Testbed.** Springy double pendulum in 3-D (Yao et al. 2021, Table 1): learn the dynamics from 500 short
trajectory chunks with a scalar HNN or a scalar N-ODE, evaluate the relative error of 150-step rollouts.

**What was done** (details in the files of this folder)
1. Reproduced the paper with the upstream code and Hamiltonian ([project_setup.md](project_setup.md)).
2. Re-implemented the bilipschitz embedder in pure JAX (thin SVD, differentiable to second order) — the earlier
   torch version could not run inside `jax.grad` ([methodology.md](methodology.md)).
3. Hyperparameter search (212 runs; depth, width, lr, activation, RBF width), selection on validation.
4. Gaussian noise on training data and on test initial conditions, with an equal-hyperparameter control.

**Results** (test rollout error, 3 seeds; full tables in [output.md](output.md) and `output/results.md`)

| | HNN inner | HNN bilip | N-ODE inner | N-ODE bilip |
|---|---|---|---|---|
| paper Table 1 | 0.005 ± 0.002 | – | 0.009 ± 0.001 | – |
| paper setting, ours | 0.0092 | 0.3210 (ReLU fails) | 0.0104 | 0.0151 |
| tuned, clean | **0.0015** | 0.0074 (SiLU) | 0.0058 | **0.0056** |
| tuned, train noise σ=0.03 | **0.0075** | 0.0151 | 0.0273 | **0.0170** |
| same config, σ = 0 / 0.03 / 0.1: bilip vs inner | | +23 % / -4 % / -32 % | | -31 % / -39 % / -15 % |

**Take-aways**
- N-ODE: the bilipschitz embedder matches the tuned baseline on clean data and lowers the error by 15-39 %
  under training noise at equal hyperparameters.
- HNN: it needs a smooth activation to train at all (ReLU + $\theta$ plateaus at 0.32) and remains ~5x worse
  than the tuned inner-product HNN on clean data, because the true Hamiltonian is linear in the inner-product
  scalars; its error grows more slowly with noise (-32 % vs the same-config baseline at σ = 0.1).
- Perturbing test initial conditions shows no embedding effect: the error is dominated by the chaotic
  sensitivity of the system itself.

**Folder map**
| file | content |
|---|---|
| [project_architecture.md](project_architecture.md) | repository tree and data flow |
| [project_setup.md](project_setup.md) · [env_setup.sh](env_setup.sh) · [data_prepare.sh](data_prepare.sh) | how to reproduce every number, compute cost |
| [dataset.md](dataset.md) | physics, sampling, splits |
| [methodology.md](methodology.md) | scalars, bilipschitz embedding, experimental protocol |
| [model_architecture.md](model_architecture.md) | notation, HNN / N-ODE, parameter counts |
| [loss.md](loss.md) · [eval_metric.md](eval_metric.md) | training loss, rollout metric |
| [output.md](output.md) | output format, result tables, conclusions |
