"""
Single entry point for the double spring pendulum experiments (Yao et al. 2021, Table 1),
with the bilipschitz embedding theta(X) = (X X^T)^{1/2} and Gaussian noise as options.

example:
    python experiments/run_dsp.py --model hnn --bilip 0 --seed 0 --out_dir results/paper_repro/hnn_inner_s0
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # experiments/ -> trainer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo -> scalaremlp

import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap
import objax
import torch
from torch.utils.data import DataLoader, Dataset
from oil.utils.utils import FixedNumpySeed, FixedPytorchSeed
from oil.datasetup.datasets import split_dataset

from scalaremlp.nn.objax import InvarianceLayer_objax, EquivarianceLayer_objax, compute_scalars, radial_basis_transform
from trainer.hamiltonian_dynamics import (IntegratedDynamicsTrainer, IntegratedODETrainer, DoubleSpringPendulum,
                                          BHamiltonianFlow, BOdeFlow, rel_err)
from trainer.utils import LoaderTo


class NoisyTrainSet(Dataset):
    """
    training split with fixed additive Gaussian noise on the observed trajectory chunks,
    noise std = noise_level * (per-coordinate std of the training states)
    """

    def __init__(self, subset, noise_level, seed):
        self.subset = subset
        Zs = np.stack([subset[i][1] for i in range(len(subset))])  # (n, chunk_len, 12)
        self.T = subset[0][0][1]
        scale = Zs.reshape(-1, Zs.shape[-1]).std(0)  # (12,)
        rng = np.random.default_rng(seed)
        self.Zs = (Zs + noise_level * scale * rng.standard_normal(Zs.shape)).astype(Zs.dtype)
        self.scale = scale

    def __len__(self):
        return self.Zs.shape[0]

    def __getitem__(self, i):
        return (self.Zs[i, 0].copy(), self.T.copy()), self.Zs[i].copy()


def rollout_error_from_noisy_ic(model_flow, ds, z0, noise, T):
    """geometric mean over (samples, time) of the relative state error between the rollout
    from the perturbed initial condition and the ground truth rollout from the clean one"""
    pred = model_flow(z0 + noise, T)
    gt = BHamiltonianFlow(ds.H, z0, T)
    errs = vmap(vmap(rel_err))(pred, gt)
    return float(jnp.exp(jnp.log(jax.lax.clamp(1e-7, errs, np.inf)).mean()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["hnn", "node"], required=True)
    parser.add_argument("--bilip", type=int, default=0, help="1: use (X X^T)^{1/2} instead of X X^T")
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_hidden", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--act", choices=["relu", "softplus", "silu", "tanh"], default="relu",
                        help="MLP activation (relu = upstream)")
    parser.add_argument("--n_rad", type=int, default=200)
    parser.add_argument("--rbf_gamma_scale", type=float, default=1.0,
                        help="N-ODE only: multiply the upstream RBF gamma (1 = upstream)")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--ndata", type=int, default=5000)
    parser.add_argument("--n_train", type=int, default=500)
    parser.add_argument("--bs", type=int, default=500)
    parser.add_argument("--data_seed", type=int, default=2021, help="upstream default, fixes the dataset and split")
    parser.add_argument("--seed", type=int, default=0, help="model init and minibatch order")
    parser.add_argument("--train_noise", type=float, default=0.0, help="relative Gaussian noise on training data")
    parser.add_argument("--eval_noise", type=str, default="0,0.01,0.03,0.1",
                        help="relative Gaussian noise on test initial conditions for the robustness evaluation")
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    t_start = time.time()

    # dataset and split exactly as upstream makeTrainerScalars
    split = {'train': args.n_train, 'val': .1, 'test': .1}
    data_config = {'chunk_len': 5, 'dt': 0.2, 'integration_time': 30, 'regen': False}
    with FixedNumpySeed(args.data_seed), FixedPytorchSeed(args.data_seed):
        base_ds = DoubleSpringPendulum(n_systems=args.ndata, **data_config)
        datasets = split_dataset(base_ds, splits=split)
    if args.train_noise > 0:
        datasets['train'] = NoisyTrainSet(datasets['train'], args.train_noise, seed=10_000 + args.seed)

    objax.random.DEFAULT_GENERATOR.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    bilip = bool(args.bilip)
    net_config = {'n_layers': args.n_layers, 'n_hidden': args.n_hidden}
    if args.model == "hnn":
        model = InvarianceLayer_objax(**net_config, bilipschitz=bilip, act=args.act)
        lr, trainer_cls = args.lr, IntegratedDynamicsTrainer
        lr_sched = lambda e: lr if (e < 200) else (lr * 0.4 if e < 1000 else (lr * 0.1))
    else:
        z0_train = np.stack([datasets['train'][i][1][0] for i in range(len(datasets['train']))])
        scalars_z0 = compute_scalars(z0_train.reshape(-1, 4, 3), bilipschitz=bilip)
        mu, gamma = radial_basis_transform(scalars_z0, nrad=args.n_rad)
        gamma = gamma * args.rbf_gamma_scale
        model = EquivarianceLayer_objax(**net_config, mu=mu, gamma=gamma, bilipschitz=bilip, act=args.act)
        lr, trainer_cls = args.lr, IntegratedODETrainer
        lr_sched = lambda e: lr if e < 300 else (lr * 0.5 if e < 1200 else lr * 0.2)

    dataloaders = {k: LoaderTo(DataLoader(v, batch_size=min(args.bs, len(v)), shuffle=(k == 'train'),
                                          num_workers=0, pin_memory=False)) for k, v in datasets.items()}
    dataloaders['Train'] = dataloaders['train']
    trainer = trainer_cls(model, dataloaders, objax.optimizer.Adam, lr_sched,
                          log_dir=None, log_args={'minPeriod': .02, 'timeFrac': .75})
    trainer.train(args.epochs)
    frame = trainer.logger.scalar_frame
    final = {k: float(v) for k, v in frame.iloc[-1].items()}
    t_train = time.time() - t_start

    # robustness: perturb clean test initial conditions, compare with ground truth of the clean ones
    test_ds = datasets['test']
    z0_test = jnp.array(np.stack([test_ds[i][0][0] for i in range(len(test_ds))]))
    scale = np.asarray(z0_test).std(0)
    if args.model == "hnn":
        flow = lambda z0, T: BHamiltonianFlow(model, z0, T)
    else:
        flow = lambda z0, T: BOdeFlow(model, z0, T)
    rng = np.random.default_rng(12345)
    eps = rng.standard_normal(z0_test.shape).astype(np.float32)
    # validation rollout error (geometric mean, T=150) for model selection, test is never used for selection
    val_ds = datasets['val']
    z0_val = jnp.array(np.stack([val_ds[i][0][0] for i in range(len(val_ds))]))
    val_rollout = rollout_error_from_noisy_ic(flow, base_ds, z0_val, 0.0, base_ds.T_long)
    robust = {}
    for lvl in [float(s) for s in args.eval_noise.split(",") if s != ""]:
        robust[str(lvl)] = rollout_error_from_noisy_ic(flow, base_ds, z0_test, jnp.array(lvl * scale * eps),
                                                       base_ds.T_long)

    result = {"args": vars(args), "final": final, "val_Rollout": val_rollout, "rollout_vs_ic_noise": robust,
              "history": {c: frame[c].dropna().tolist() for c in frame.columns},
              "history_step": frame.index.tolist(),
              "nonfinite": bool(not np.isfinite(list(final.values())).all()),
              "train_seconds": t_train, "total_seconds": time.time() - t_start,
              "device": str(jax.devices()[0])}
    with open(os.path.join(args.out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=1)
    objax.io.save_var_collection(os.path.join(args.out_dir, "model.npz"), model.vars())
    print(json.dumps({"final": final, "val_Rollout": val_rollout, "rollout_vs_ic_noise": robust, "seconds": result["total_seconds"]}))


if __name__ == "__main__":
    main()
