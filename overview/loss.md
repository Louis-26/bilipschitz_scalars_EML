# Loss

Both models are trained with the same trajectory-matching loss (`IntegratedDynamicsTrainer.loss`,
`IntegratedODETrainer.loss`): starting from the observed initial state of each training chunk, the learned
dynamics are integrated over the chunk's 5 time points and compared with the observed states,

$$\mathcal L(\theta)=\frac{1}{B\cdot 5\cdot 12}\sum_{b=1}^{B}\sum_{t=0}^{4}\big\lVert \hat z_\theta^{(b)}(t_k)-z^{(b)}(t_k)\big\rVert_2^2,
\qquad \hat z_\theta^{(b)}=\mathrm{odeint}\big(f_\theta,\ z^{(b)}(0),\ [0,0.2,\dots,0.8]\big)$$

with $f_\theta = J\nabla H_\theta$ (HNN) or the equivariant N-ODE field. $B=500$ (full batch). Gradients flow
through `odeint` by its adjoint method; for the HNN this means second derivatives of $H_\theta$ w.r.t. $z$,
which is why the bilipschitz embedding must be twice differentiable (see methodology.md).

With training noise, $z^{(b)}$ (both the initial state and the targets) are the noisy observations.

The same quantity on the val/test splits is logged as `val_MSE` / `test_MSE`; `Train_MSE` is computed on the
(possibly noisy) training split.
