# Model architecture and notation

## Notation
| symbol | meaning | shape |
|---|---|---|
| $z=(q_1,q_2,p_1,p_2)$ | state (positions, momenta of the two masses; origin $q_0=0$) | $\mathbb R^{12}$ |
| $X$ | $z$ reshaped, rows $u_1..u_4 = q_1,q_2,p_1,p_2$ | $\mathbb R^{4\times3}$ |
| $g=(0,0,-1)$ | gravity direction (input vector) | $\mathbb R^3$ |
| $y=q_1-q_2$ | second-spring vector | $\mathbb R^3$ |
| $s(z)$ | 30 scalars (`compute_scalars_jax`) | $\mathbb R^{30}$ |
| $\theta(X)=(XX^\top)^{1/2}$ | bilipschitz embedding, replaces $XX^\top$ in $s$ when `bilipschitz=True` | $\mathbb R^{4\times4}$ |
| $L,\ h$ | `n_layers` (extra hidden layers), `n_hidden` (width) | |

## MLP (`BasicMLP_objax`)
`Linear(n_in, h) → ReLU → [Linear(h, h) → ReLU] × L → Linear(h, n_out)`, objax default init.
With the paper setting ($L=3$, $h=100$) the network has 5 linear layers.

## HNN (`InvarianceLayer_objax`)
$$H_\theta(z)=\mathrm{MLP}_\theta(s(z))\in\mathbb R,\qquad \dot z = J\nabla_z H_\theta(z)$$
`n_in = 30`, `n_out = 1`. The gradient is taken by `jax.grad` inside `odeint`.

## N-ODE (`EquivarianceLayer_objax`)
$$\phi(s)_{k,r}=\exp(-\gamma (s_k-\mu_r)^2),\ r=1..200,\qquad c=\mathrm{MLP}_\theta(\phi(s))\in\mathbb R^{24}$$
$$\dot u_i=\sum_{j=1}^4 c_{4(i-1)+j}\,u_j + c_{16+i}\,y + c_{20+i}\,g$$
`n_in = 30 × 200 = 6000`, `n_out = 24`. $\mu$ = 200 equispaced centres between the min and max of all training
scalars, $\gamma = 2(\max-\min)/199$ (computed with the same embedding as the model).

## Trainable parameters
| config (L-h) | HNN | N-ODE |
|---|---|---|
| 3-100 (paper) | 33,501 | 632,824 |
| 5-150 | 118,051 | 1,017,024 |
| 7-200 | 287,801 | 1,486,424 |

The bilipschitz variant has exactly the same parameters (the embedding has none).

## Optimisation
Adam (objax), full batch (500), 2000 epochs, step schedules from upstream:
- HNN: lr for epochs < 200, 0.4·lr until 1000, 0.1·lr after.
- N-ODE: lr for epochs < 300, 0.5·lr until 1200, 0.2·lr after.
