# Evaluation metrics

## Rollout error (main metric, the paper's Table 1)
For each test initial condition $z_0$, the learned model and the true Hamiltonian are both integrated over
$T=150$ steps ($t\in\{0,0.2,\dots,29.8\}$):

$$\mathrm{err}(t)=\frac{\sqrt{\operatorname{mean}\big((\hat z(t)-z(t))^2\big)}}{\sqrt{\operatorname{mean}(\hat z(t)^2)}+\sqrt{\operatorname{mean}(z(t)^2)}}\in[0,1]$$

$$\texttt{test\_Rollout}=\exp\Big(\frac{1}{N\,T}\sum_{n,t}\log\max(\mathrm{err}_n(t),10^{-7})\Big)$$

i.e. the geometric mean of the state relative error over the 500 test trajectories and 150 steps
(`log_rollout_error[_ode]` in `trainer/hamiltonian_dynamics.py`). Lower is better. Paper: HNN 0.005 ± 0.002,
N-ODE 0.009 ± 0.001 (3 trials).

## Other metrics
| name | definition | use |
|---|---|---|
| `val_Rollout` | same as above on the 500 validation initial conditions, after training | **model selection** |
| `test_MSE`, `val_MSE`, `Train_MSE` | training loss (5-step chunks) on the split | diagnostics |
| `rollout_vs_ic_noise[σ]` | rollout error from $z_0+\sigma s\,\epsilon$ against the true rollout from clean $z_0$ | robustness to input noise |

## Reporting
- Upstream reports the last logged row of the training log (not the best epoch); we do the same (`final`).
- Seeds: mean ± std over seeds 0, 1, 2 (model initialisation and batch order; the dataset and split are fixed).
- Selection of hyperparameters uses `val_Rollout` only; test numbers are reported for the selected configs.
