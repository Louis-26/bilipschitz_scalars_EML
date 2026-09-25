# Dataset: springy double pendulum (Finzi et al. 2021; Yao et al. 2021, Sec. 5)

Two point masses connected in series by two springs, the first spring attached to the origin, under uniform
gravity. The data are simulated, nothing is downloaded (`overview/data_prepare.sh`).

## Physics
State $z = (q_1, q_2, p_1, p_2) \in (\mathbb R^3)^4$, flattened to $\mathbb R^{12}$ in this order.
All constants are 1: $m_1=m_2=k_1=k_2=l_1=l_2=g=1$, gravity $g=(0,0,-1)$.

$$H(q,p)=\frac{|p_1|^2}{2m_1}+\frac{|p_2|^2}{2m_2}
+\tfrac12 k_1(|q_1|-l_1)^2 + k_2(|q_1-q_2|-l_2)^2 + m_1 g\,q_{1,z} + m_2 g\,q_{2,z}$$

The second spring term has no factor 1/2: this is the upstream code (`emlp`/`ScalarEMLP`) that produced the
paper's numbers, and it is what `trainer/hamiltonian_dynamics.py` uses again (see `work_progress/code_change.md`).
The trajectories follow $\dot z = J\nabla H(z)$, $J=\begin{psmallmatrix}0&I\\-I&0\end{psmallmatrix}$, integrated
with `jax.experimental.ode.odeint` (Dormand-Prince, rtol = atol = 1e-4).

## Sampling
| quantity | distribution |
|---|---|
| $q_1(0)$ | $\mathcal N((0,0,-1.5),\,0.2^2 I)$ |
| $q_2(0)$ | $\mathcal N((0,0,-3),\,0.2^2 I)$ |
| $p_1(0),p_2(0)$ | $\mathcal N(0,\,0.4^2 I)$ |

Each of the 5000 systems is integrated for $T=30$ with $\Delta t=0.2$ (150 steps); the 150 steps are cut into
30 chunks of `chunk_len = 5` steps and one chunk is kept at random. The initial condition of a sample is the
first state of its chunk.

## Stored tensor and splits
- `ScalarEMLP_bilipschitz/datasets/ODEDynamics/DoubleSpringPendulum/trajectories_5000_5_0.2_30.pt`:
  numpy float32 array `(5000, 5, 12)`, generated under `FixedNumpySeed(2021)`.
- `split_dataset(seed 2021)`: **train 500, val 500 (10 %), test 500 (10 %)**, the remaining 3500 are unused
  (upstream default `split={'train':500,'val':.1,'test':.1}`).
- A sample is `((z0, T), z)` with `z0` (12,), `T = [0, .2, .4, .6, .8]`, `z` (5, 12).
- Evaluation rollouts are not stored: they are re-integrated from the test initial conditions with the true
  $H$ over `T_long = arange(0, 30, 0.2)` (150 steps).

Per-coordinate std of the stored states: positions 0.47-1.18, momenta 0.38-0.78 (these scales define the
relative Gaussian noise of the robustness study).

## Earlier caches
The older caches in `datasets/` (repo root), `ScalarEMLP_bilipschitz/experiments/datasets/` and
`.../parameter_tuning/datasets/` were generated with a modified Hamiltonian ($\tfrac12 k_2$ instead of $k_2$;
checked by re-integration: error 7e-5 with $\tfrac12 k_2$, 0.65 with $k_2$). They are not used by this study.
