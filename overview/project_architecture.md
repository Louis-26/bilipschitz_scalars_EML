# Project architecture

```
📁 bilipschitz_scalars_EML/
├── 📁 overview/                          # this folder: abstract, setup, dataset, method, model, loss, metric, output
├── 📁 ScalarEMLP_bilipschitz/            # fork of github.com/weichiyao/ScalarEMLP with the bilipschitz embedding
│   ├── 📁 scalaremlp/                    # python package (installed through PYTHONPATH, see project_setup.md)
│   │   ├── 📁 nn/
│   │   │   └── 📄 objax.py               # ★ scalars + models: gram_sqrt[_jax], comp_inner_products[_jax],
│   │   │                                 #   compute_scalars[_jax], InvarianceLayer_objax (HNN), EquivarianceLayer_objax (N-ODE)
│   │   ├── 📁 reps/ 📄 groups.py         # EMLP representation/group code imported by upstream (not used by the scalar models)
│   │   └── 📄 datasets.py                # O(5), inertia, particle scattering datasets (not used here)
│   ├── 📁 experiments/
│   │   ├── 📄 run_dsp.py                 # ★ single entry point: model x embedding x hyperparameters x noise -> result.json
│   │   ├── 📄 prepare_data.py            # ★ one-time generation of the pendulum dataset (seed 2021)
│   │   ├── 📄 make_tasks.py              # ★ writes the task lists of each stage (paper / tune / final / noise)
│   │   ├── 📄 summarize.py               # ★ result.json -> output/results.md, output/noise_robustness.png
│   │   ├── 📁 trainer/
│   │   │   ├── 📄 hamiltonian_dynamics.py# DoubleSpringPendulum dataset, HNN/N-ODE trainers, rollout metric
│   │   │   ├── 📄 trainer.py             # base training loop (objax), logging
│   │   │   ├── 📄 classifier.py          # Regressor base class
│   │   │   └── 📄 utils.py               # torch loader -> jax arrays
│   │   ├── 📄 hnn_scalars.py 📄 neuralode_scalars.py   # upstream entry points (kept, not used by the study)
│   │   ├── 📁 parameter_tuning/          # earlier tuning scripts and results (0.5*k2 Hamiltonian, see code_change.md)
│   │   └── 📁 data_add_noise/            # earlier noise stubs (superseded by run_dsp.py --train_noise)
│   ├── 📁 datasets/ODEDynamics/DoubleSpringPendulum/trajectories_5000_5_0.2_30.pt   # ★ dataset cache (upstream H)
│   └── 📁 test/                          # exploratory notebooks
├── 📁 experiment/                        # ★ one folder per run: result.json, model.npz, log.txt
│   ├── 📁 paper/                         #   paper setting, inner vs bilip, seeds 0-2
│   ├── 📁 tune/                          #   27-point grid (seed 0) + seeds 1-2 of the selected configs
│   └── 📁 noise/                         #   Gaussian noise on training data
├── 📁 output/                            # ★ results.md, best_configs.json, noise_robustness.png
├── 📁 SLURM_execution/
│   ├── 📁 SLURM_script/                  # run_tasks.sh (job array runner), gpu_check.sh, templates, hpc_utils.sh
│   ├── 📁 SLURM_output/                  # .out/.err of every job
│   └── 📁 SLURM_outcome/                 # time logs, tasks/*.txt (the exact argument lines that were run)
├── 📁 work_progress/                     # code_change.md and stage notes
├── 📁 literature_reference/              # Yao 2021, Villar 2021, Balan & Dock 2022, ...
├── 📁 scalar_mlp_env/                    # exported conda environments (cpu / gpu)
└── 📄 bilipschitz_test.ipynb             # numerical check of the Balan-Dock bounds for theta(X) = (X X^T)^{1/2}
```

★ = files that the reproduction in this folder relies on.

## Data flow of one run

```
prepare_data.py ──> datasets/.../trajectories_5000_5_0.2_30.pt  (5000 chunks x 5 steps x 12)
                                   │
run_dsp.py ── split (train 500 / val 500 / test 500, seed 2021) ── optional NoisyTrainSet (train only)
     │
     ├── HNN : z ─> scalars(30) ─> MLP ─> H(z) ─> dz/dt = J ∇H ─> odeint
     └── N-ODE: z ─> scalars(30) ─> RBF(200) ─> MLP ─> 24 coefficients ─> equivariant dz/dt ─> odeint
     │
     └── 2000 epochs Adam ─> result.json (final metrics, val rollout, IC-noise robustness, history) + model.npz
```
