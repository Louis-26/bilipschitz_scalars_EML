# Work stage 1 (2026-09-23 / 24): reproduction, bilipschitz embedder, Gaussian noise

## Finished
- Audit of the repo: the torch bilipschitz code could not run under jax (`TypeError` eagerly, tracer error
  under `grad`); earlier "with embedding" tables came from another checkout; the branch Hamiltonian had
  `.5*k2` (all old caches made with it). See `code_change.md`.
- Paper reproduction with the upstream Hamiltonian: HNN 0.0092 ± 0.0005 (paper 0.005 ± 0.002),
  N-ODE 0.0104 ± 0.0011 (paper 0.009 ± 0.001).
- Pure-JAX bilipschitz embedder (thin SVD), gradient-checked; activation option (bilip HNN needs a smooth one).
- Search (212 runs), 3-seed confirmation, Gaussian training noise (36), equal-config control (22).
  Results: `overview/output.md`, `output/results.md`, `output/noise_robustness.png`.

## Findings
- N-ODE: bilip = tuned baseline on clean data (0.0056 vs 0.0058), 15-39 % lower error under training noise
  at equal hyperparameters.
- HNN: bilip 0.0074 vs tuned baseline 0.0015 on clean data; less degradation with noise (-32 % vs the
  same-config baseline at σ = 0.1). The inner-product scalars make the true H linear, the square root does not.
- Test-IC perturbation: no embedding effect (dominated by the system's own sensitivity).

## Possible next steps
- Adversarial attack on initial conditions / training data (README task 4), reusing `run_dsp.py`.
- Bilipschitz variants that keep the linear structure: e.g. add the squared norms back, or apply θ only
  to the momenta / positions block; per-block RBF grids for the N-ODE.
- More data seeds (vary `--data_seed`) to separate data variance from seed variance.

## Compute
290 runs, 35.5 GPU-hours / 532 SU (from tune3 on only l40s, 9 SU/h per job).
