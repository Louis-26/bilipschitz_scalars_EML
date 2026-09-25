"""
write task files (one run_dsp.py argument line per run) for the stages of the study

stage paper : paper setting (3 layers, 100 hidden, lr 5e-3), inner product vs bilipschitz, seeds 0-2
stage tune  : 3x3x3 grid over (n_layers, n_hidden, lr) for both embeddings, seed 0
stage probe : smooth activations for the HNN, RBF width for the N-ODE, seed 0
stage tune2 : HNN grid with smooth activations (silu, softplus), both embeddings, seed 0
stage tune3 : larger learning rates / widths where tune2 hit the boundary, both embeddings, seed 0
stage final : best tune/probe config (by val rollout) per (model, embedding), seeds 1-2 on top of seed 0
stage noise : training noise levels for the selected configs, seeds 0-2
stage control : inner product with the bilipschitz-selected hyperparameters, clean seeds 1-2 + noise

usage: python experiments/make_tasks.py STAGE
"""
import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
EXP = os.path.join(REPO, "experiment")
TASKS = os.path.join(REPO, "SLURM_execution", "SLURM_outcome", "tasks")
PAPER = (3, 100, 5e-3)
GRID = list(itertools.product([3, 5, 7], [100, 150, 200], [1e-2, 5e-3, 3e-3]))
NOISE = [0.01, 0.03, 0.1]
EMB = {0: "inner", 1: "bilip"}


def line(stage, model, bilip, cfg, seed, noise=0.0, act="relu", gscale=1.0):
    """cfg = (n_layers, n_hidden, lr); act / gscale (RBF gamma scale) only appear in the name when non-default"""
    nl, nh, lr = cfg
    extra = ("" if act == "relu" else f"_{act}") + ("" if gscale == 1.0 else f"_g{gscale:g}")
    name = f"{model}_{EMB[bilip]}_L{nl}_H{nh}_lr{lr:g}{extra}_noise{noise:g}_s{seed}"
    return (f"--model {model} --bilip {bilip} --n_layers {nl} --n_hidden {nh} --lr {lr:g} --act {act} "
            f"--rbf_gamma_scale {gscale:g} --seed {seed} --train_noise {noise:g} --out_dir {os.path.join(EXP, stage, name)}")


def variant(args):
    """full identity of a configuration (everything except seed / noise / out_dir)"""
    return (args["model"], args["bilip"], args["n_layers"], args["n_hidden"], args["lr"],
            args.get("act", "relu"), args.get("rbf_gamma_scale", 1.0))


def best_configs():
    """best clean seed-0 configuration per (model, embedding) among the grid (tune) and probe runs,
    by validation rollout error; returns {(model, bilip): (cfg, act, gscale)}"""
    import glob
    best = {}
    for f in sum((glob.glob(os.path.join(EXP, st, "*", "result.json")) for st in ("tune", "probe", "tune2", "tune3")), []):
        r = json.load(open(f))
        a = r["args"]
        if a["seed"] != 0 or a["train_noise"] != 0 or r["nonfinite"] or not r["val_Rollout"] < 1:
            continue
        v = variant(a)
        k = v[:2]
        if k not in best or r["val_Rollout"] < best[k][0]:
            best[k] = (r["val_Rollout"], v)
    return {k: ((v[2], v[3], v[4]), v[5], v[6]) for k, (_, v) in best.items()}


if __name__ == "__main__":
    stage = sys.argv[1]
    lines = []
    if stage == "paper":
        for model, bilip, seed in itertools.product(["hnn", "node"], [0, 1], [0, 1, 2]):
            lines.append(line("paper", model, bilip, PAPER, seed))
    elif stage == "tune":
        for model, bilip, cfg in itertools.product(["hnn", "node"], [0, 1], GRID):
            lines.append(line("tune", model, bilip, cfg, 0))
    elif stage == "probe":
        # HNN: smooth activations (a ReLU-MLP Hamiltonian has a piecewise-constant gradient field)
        for bilip, act, lr in itertools.product([0, 1], ["softplus", "silu", "tanh"], [1e-2, 5e-3]):
            lines.append(line("probe", "hnn", bilip, (3, 100, lr), 0, act=act))
        # N-ODE: RBF width (the bilipschitz scalars shrink the global RBF range and hence gamma by 2.44x)
        for bilip, g, lr in itertools.product([0, 1], [2.44, 4.0], [1e-2, 5e-3]):
            lines.append(line("probe", "node", bilip, (3, 100, lr), 0, gscale=g))
    elif stage == "tune2":
        # second HNN round with smooth activations, both embeddings
        for bilip, act, nl, nh, lr in itertools.product([0, 1], ["silu", "softplus"], [3, 5], [100, 200], [2e-2, 1e-2, 5e-3]):
            lines.append(line("tune2", "hnn", bilip, (nl, nh, lr), 0, act=act))
    elif stage == "tune3":
        # extend the boundaries hit in tune/tune2 (largest lr), for both embeddings
        for act, nl, nh, lr in itertools.product(["silu", "softplus"], [3, 5], [200, 300], [2e-2, 3e-2, 5e-2]):
            lines.append(line("tune3", "hnn", 1, (nl, nh, lr), 0, act=act))
        for nl, nh, lr in itertools.product([3, 5], [100, 200], [2e-2, 3e-2]):
            lines.append(line("tune3", "hnn", 0, (nl, nh, lr), 0))
        for nl, nh in itertools.product([5, 7], [150, 200]):
            lines.append(line("tune3", "node", 0, (nl, nh, 2e-2), 0))
        # the two runs killed after hanging in the step-0 rollout (before the solver step cap)
        lines.append(line("paper", "node", 1, PAPER, 0))
        lines.append(line("tune", "node", 1, PAPER, 0))
    elif stage == "final":
        for (model, bilip), (cfg, act, g) in sorted(best_configs().items()):
            print(model, EMB[bilip], cfg, act, g)
            for seed in [1, 2]:
                lines.append(line("final", model, bilip, cfg, seed, act=act, gscale=g))
    elif stage == "noise":
        best = best_configs()
        for model, bilip, noise, seed in itertools.product(["hnn", "node"], [0, 1], NOISE, [0, 1, 2]):
            cfg, act, g = best[(model, bilip)]  # selected configs only (the ReLU paper config fails for bilip HNN)
            lines.append(line("noise", model, bilip, cfg, seed, noise, act=act, gscale=g))
    elif stage == "control":
        # inner-product models with exactly the hyperparameters selected for the bilipschitz models,
        # to separate the effect of the embedding from that of the hyperparameters under noise
        best = best_configs()
        for model in ["hnn", "node"]:
            cfg, act, g = best[(model, 1)]
            for seed in [1, 2]:
                lines.append(line("control", model, 0, cfg, seed, act=act, gscale=g))
            for noise, seed in itertools.product(NOISE, [0, 1, 2]):
                lines.append(line("control", model, 0, cfg, seed, noise, act=act, gscale=g))
    os.makedirs(TASKS, exist_ok=True)
    path = os.path.join(TASKS, f"{stage}.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(path, len(lines))
