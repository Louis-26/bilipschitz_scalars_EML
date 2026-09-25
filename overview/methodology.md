# Methodology

## 1. Scalar-based equivariant models (Villar et al. 2021; Yao et al. 2021)
By the first fundamental theorem of invariant theory for $O(d)$, a function of vectors $v_1,\dots,v_n$ is
$O(d)$-invariant iff it is a function of the inner products $v_i^\top v_j$, and an $O(d)$-equivariant vector
function can be written as $h(V)=\sum_t f_t(V)\,v_t$ with invariant $f_t$. Gravity breaks $O(3)$ down to
$O(2)$; the scalar method restores full $O(3)$ equivariance by passing $g$ as an extra input vector.

For the pendulum the vectors are $u = (q_1, q_2, p_1, p_2)$ plus $g$ and $q_1-q_2$, and the 30 scalars are

| block | entries | count |
|---|---|---|
| norms | $\lVert u_i\rVert$ | 4 |
| Gram | $(XX^\top)_{ij}=u_i^\top u_j$, $X=[u_1;\dots;u_4]\in\mathbb R^{4\times3}$ | 16 |
| gravity | $g^\top u_i$ | 4 |
| spring 2 | $\lVert q_1-q_2\rVert^2,\ \lVert q_1-q_2\rVert$ | 2 |
| spring 2 | $(q_1-q_2)^\top u_i$ | 4 |

- **HNN**: an MLP maps the scalars to $H_\theta(z)$; the dynamics $J\nabla_z H_\theta$ are equivariant and
  symplectic by construction.
- **N-ODE**: each scalar is expanded in 200 Gaussian radial basis functions, an MLP outputs 24 invariant
  coefficients, and $\dot z$ is assembled equivariantly: $\dot u_i=\sum_j c_{ij}u_j+c^{(y)}_i(q_1-q_2)+c^{(g)}_i g$.

## 2. Bilipschitz embedding (this project)
The Gram block is invariant but not bilipschitz: $X\mapsto XX^\top$ is quadratic, so the map to the quotient
$\mathbb R^{4\times3}/O(3)$ has no global lower Lipschitz bound near 0 and no upper bound at infinity.
Balan & Dock (2022, Thm 2.7) show that
$$\theta(X) = (XX^\top)^{1/2}$$
satisfies $D(X,Y)\le\lVert\theta(X)-\theta(Y)\rVert_F\le\sqrt2\,D(X,Y)$ with the Procrustes distance
$D(X,Y)=\min_{U\in O(3)}\lVert X-YU\rVert_F$ (checked numerically in `bilipschitz_test.ipynb`: ratios in
[1.04, 1.13] for random Gaussian pairs). The **bilipschitz variant replaces the 16 Gram entries by the entries
of $\theta(X)$**; everything else (the other 14 scalars, networks, training) is unchanged, so the comparison
isolates the embedding. Note that only this block becomes bilipschitz; the full 30-vector is not.

### Computation
$XX^\top$ is a $4\times4$ matrix of rank $\le 3$, so its square root through an eigen/SVD of $XX^\top$ hits
$\sqrt{0}$, whose derivative is infinite, and HNN training needs second derivatives (∇H inside the ODE, then
backpropagation through the solver). We use the thin SVD of $X$ itself:
$$X=U\Sigma V^\top\ \Rightarrow\ (XX^\top)^{1/2}=U\Sigma U^\top,\qquad U\in\mathbb R^{4\times3}$$
(`gram_sqrt_jax` in `scalaremlp/nn/objax.py`), which is exact, $O(3)$-invariant and differentiable wherever the
singular values of $X$ are distinct. On all 25 000 states of the dataset the float32 gradients agree with float64
to 8e-6 (relative, worst of the 200 most degenerate states) and Hessian-vector products to 3.5e-3.
Newton-Schulz iterations were tested and rejected (diverge in float32 because of the zero eigenvalue).

## 3. Experimental protocol
All runs: 2000 epochs, full batch (500), upstream learning-rate step schedules, dataset and split fixed (seed 2021);
`seed` = model initialisation and batch order. Selection always uses the **validation** rollout; test numbers are
only reported.

| stage | runs | what |
|---|---|---|
| `paper` | 12 | upstream setting (3 hidden layers × 100, ReLU, lr 5e-3), inner vs bilip, HNN and N-ODE, seeds 0-2 |
| `tune` | 108 | grid layers {3,5,7} × width {100,150,200} × lr {1e-2, 5e-3, 3e-3}, both embeddings and models, seed 0 |
| `probe` | 20 | HNN activations {softplus, silu, tanh}; N-ODE RBF width γ × {2.44, 4} (2.44 = ratio of the RBF γ of the two embeddings) |
| `tune2` | 48 | HNN with smooth activations {silu, softplus} × layers {3,5} × width {100,200} × lr {2e-2, 1e-2, 5e-3}, both embeddings |
| `tune3` | 38 | larger lr / width where the best configs sat on the grid boundary (HNN lr up to 5e-2, width 300; N-ODE inner lr 2e-2) |
| `final` | 8 | the selected config of each (model, embedding), seeds 1-2 (seed 0 comes from the search) |
| `noise` | 36 | selected configs trained on noisy data, σ ∈ {0.01, 0.03, 0.1}, seeds 0-2 |
| `control` | 22 | inner product **with the bilip-selected hyperparameters**, clean (seeds 1-2) and noisy (σ × seeds 0-2) |

Why the extra stages:
- **HNN activation.** With ReLU the bilipschitz HNN plateaus at rollout error ≈ 0.32 for every width, depth
  and learning rate (train MSE stuck at 1.25e-2), while its gradients are verified correct (float64 finite
  differences agree to 3e-9). The HNN vector field is $J\nabla H$; a ReLU-MLP $H$ is piecewise linear in the
  scalars, so $\nabla H$ is piecewise constant in feature space. The upstream scalars make the true $H$ exactly
  linear ($\tfrac12|p_i|^2$ are Gram entries, $\tfrac12(|q_1|-1)^2=\tfrac12|q_1|^2-|q_1|+\tfrac12$,
  $|q_1-q_2|$ and $|q_1-q_2|^2$ are both inputs, gravity is $g^\top q_i$), so ReLU is harmless there;
  in the bilipschitz scalars $|p_i|^2$ is quadratic, which a ReLU $H$ cannot differentiate correctly.
  Smooth activations remove the plateau.
- **N-ODE RBF width.** All 30 scalars share one grid of 200 RBF centres between the global min and max of the
  training scalars; the Gram block sets the max (41 for $XX^\top$, 5.5 for $(XX^\top)^{1/2}$), so the
  embedding also changes the RBF grid and γ. Rescaling γ did not help (kept at the upstream value).
- **Control.** The selected configs differ between embeddings (e.g. N-ODE lr 2e-2 vs 5e-3, HNN ReLU vs SiLU);
  training the inner-product model with the bilip hyperparameters separates the embedding from the
  hyperparameters in the noise comparison.

### Gaussian noise
- **Training noise**: fixed additive noise $\epsilon\sim\mathcal N(0,(\sigma s_c)^2)$ on every observed training
  state (initial condition and targets), $s_c$ the per-coordinate std of the training states,
  $\sigma\in\{0.01,0.03,0.1\}$; validation and test data stay clean.
- **Perturbed test initial conditions**: every trained model is also rolled out from $z_0+\sigma s\,\epsilon$
  ($\sigma\in\{0.01,0.03,0.1\}$, one fixed $\epsilon$) and compared with the true rollout from the clean $z_0$.
