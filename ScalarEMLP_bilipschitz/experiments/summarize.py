"""
collect experiment/*/*/result.json into markdown tables and a figure under output/

usage: python experiments/summarize.py
"""
import glob
import itertools
import json
import os
from collections import defaultdict

import numpy as np

from make_tasks import EXP, PAPER, best_configs, variant

REPO = os.path.abspath(os.path.join(EXP, ".."))
OUT = os.path.join(REPO, "output")
PAPER_TABLE1 = {"hnn": (0.005, 0.002), "node": (0.009, 0.001)}
EMB = {0: "inner", 1: "bilip"}
NAME = {"hnn": "HNN", "node": "N-ODE", 0: "inner $XX^T$", 1: "bilip $(XX^T)^{1/2}$"}
IC_LEVELS = ["0.0", "0.01", "0.03", "0.1"]


def load(*stages):
    runs = []
    for stage in stages:
        for f in sorted(glob.glob(os.path.join(EXP, stage, "*", "result.json"))):
            r = json.load(open(f))
            r["stage"] = stage
            r["variant"] = variant(r["args"])
            r["ok"] = (not r["nonfinite"]) and np.isfinite(r["val_Rollout"]) and r["final"]["test_Rollout"] < 1
            runs.append(r)
    return runs


def by(runs, noise=0.0):
    """group runs by variant for one training-noise level"""
    g = defaultdict(list)
    for r in runs:
        if r["args"]["train_noise"] == noise:
            g[r["variant"]].append(r)
    return g


def dedup_seeds(rs):
    """one run per seed (the paper config seed 0 exists in both paper/ and tune/): prefer paper > final > others"""
    prio = {"paper": 0, "final": 1, "noise": 1}
    out = {}
    for r in sorted(rs, key=lambda r: prio.get(r["stage"], 2)):
        out.setdefault(r["args"]["seed"], r)
    return [out[s] for s in sorted(out)]


def ms(vals, digits=4):
    v = np.asarray(vals, dtype=float)
    if len(v) == 0:
        return "n/a"
    return f"{v.mean():.{digits}f} ± {v.std():.{digits}f}" if len(v) > 1 else f"{v.mean():.{digits}f}"


def vstr(v):
    s = f"{v[2]}-{v[3]}-{v[4]:g}"
    if v[5] != "relu":
        s += f", {v[5]}"
    if v[6] != 1.0:
        s += f", γ×{v[6]:g}"
    return s


def row(v, rs, extra=""):
    ok = [r for r in rs if r["ok"]]
    div = len(rs) - len(ok)
    return (f"| {NAME[v[0]]} | {NAME[v[1]]} | {vstr(v)}{extra} | {len(rs)}{f' ({div} diverged)' if div else ''} | "
            f"{ms([r['val_Rollout'] for r in ok])} | {ms([r['final']['test_Rollout'] for r in ok])} | "
            f"{ms([r['final']['test_MSE'] for r in ok], 6)} |")


HEAD = ["| model | features | config (L-h-lr) | runs | val rollout | test rollout | test MSE |",
        "|---|---|---|---|---|---|---|"]


def paper_section(clean):
    rows = list(HEAD)
    for model, bilip in itertools.product(["hnn", "node"], [0, 1]):
        v = (model, bilip) + PAPER + ("relu", 1.0)
        rs = [r for r in clean[v] if r["stage"] == "paper"]
        if rs:
            p = PAPER_TABLE1[model]
            rows.append(row(v, rs) + f" paper: {p[0]:.3f} ± {p[1]:.3f} |")
    rows[0] += " paper Table 1 |"
    rows[1] += "---|"
    return "\n".join(rows)


def grid_section(runs):
    out = []
    for model in ["hnn", "node"]:
        rows = [f"**{NAME[model]}** (seed 0, clean data, sorted by the better of the two val rollouts)", "",
                "| config (L-h-lr) | inner: val | inner: test | bilip: val | bilip: test |", "|---|---|---|---|---|"]
        cells = defaultdict(dict)
        for r in runs:
            a = r["args"]
            if a["model"] != model or a["seed"] != 0 or a["train_noise"] != 0:
                continue
            cells[r["variant"][2:]][a["bilip"]] = r
        fmt = lambda r, k: "n/a" if r is None else ("diverged" if not r["ok"] else
                                                     f"{(r['val_Rollout'] if k == 'val' else r['final']['test_Rollout']):.4f}")
        key = lambda c: min([cells[c][b]["val_Rollout"] for b in cells[c] if cells[c][b]["ok"]] or [9])
        for c in sorted(cells, key=key):
            ri, rb = cells[c].get(0), cells[c].get(1)
            rows.append(f"| {vstr((model, 0) + c)} | {fmt(ri, 'val')} | {fmt(ri, 'test')} | {fmt(rb, 'val')} | {fmt(rb, 'test')} |")
        out.append("\n".join(rows))
    return "\n\n".join(out)


def final_section(clean, best):
    rows = list(HEAD)
    for model, bilip in itertools.product(["hnn", "node"], [0, 1]):
        vp = (model, bilip) + PAPER + ("relu", 1.0)
        rows.append(row(vp, [r for r in clean[vp] if r["stage"] == "paper"], " (paper)"))
        if (model, bilip) in best:
            cfg, act, g = best[(model, bilip)]
            vb = (model, bilip) + cfg + (act, g)
            rs = dedup_seeds([r for r in clean[vb] if r["stage"] in ("tune", "probe", "tune2", "tune3", "final")])
            rows.append(row(vb, rs, " (**selected**)"))
    return "\n".join(rows)


def noise_section(runs, best):
    levels = sorted({r["args"]["train_noise"] for r in runs})
    head = ("| model | features | config | " + " | ".join(f"σ_train={l:g}" for l in levels) + " |\n"
            "|---|---|---|" + "---|" * len(levels))
    rows, series = [head], {}
    for model, bilip in itertools.product(["hnn", "node"], [0, 1]):
        if (model, bilip) not in best:
            continue
        for cfg, act, g in [best[(model, bilip)]]:
            v = (model, bilip) + cfg + (act, g)
            cells, ys = [], []
            for l in levels:
                rs = dedup_seeds([r for r in by(runs, l)[v] if r["ok"]])
                vals = [r["final"]["test_Rollout"] for r in rs]
                cells.append(ms(vals) if vals else "n/a")
                ys.append((np.mean(vals), np.std(vals)) if vals else (np.nan, np.nan))
            series[v] = (levels, ys)
            rows.append(f"| {NAME[model]} | {NAME[bilip]} | {vstr(v)} | " + " | ".join(cells) + " |")
    return "\n".join(rows), series


def control_section(all_runs, best):
    """same hyperparameters for both embeddings: inner product with the bilip-selected config (stage control)"""
    levels = [0.0, 0.01, 0.03, 0.1]
    head = ("| model | features | config | " + " | ".join(f"σ_train={l:g}" for l in levels) + " | bilip vs inner, same config |\n"
            "|---|---|---|" + "---|" * (len(levels) + 1))
    rows, series = [head], {}
    for model in ["hnn", "node"]:
        if (model, 1) not in best:
            continue
        cfg, act, g = best[(model, 1)]
        vals = {}
        for bilip in [0, 1]:
            v = (model, bilip) + cfg + (act, g)
            ys = []
            for l in levels:
                rs = dedup_seeds([r for r in by(all_runs, l)[v] if r["ok"]])
                ys.append([r["final"]["test_Rollout"] for r in rs])
            vals[bilip] = ys
            series[v] = (levels, [(np.mean(y), np.std(y)) if y else (np.nan, np.nan) for y in ys])
        rel = " / ".join(f"{100 * (np.mean(b) / np.mean(i) - 1):+.0f}%" if b and i else "n/a"
                         for i, b in zip(vals[0], vals[1]))
        for bilip in [0, 1]:
            v = (model, bilip) + cfg + (act, g)
            rows.append(f"| {NAME[model]} | {NAME[bilip]} | {vstr(v)} | " + " | ".join(ms(y) for y in vals[bilip]) +
                        f" | {rel if bilip else ''} |")
    return "\n".join(rows), series


def ic_section(clean, best):
    rows = ["| model | features | config | " + " | ".join(f"σ_IC={l}" for l in IC_LEVELS) + " | ratio σ=0.1 / clean |",
            "|---|---|---|" + "---|" * (len(IC_LEVELS) + 1)]
    for model, bilip in itertools.product(["hnn", "node"], [0, 1]):
        cands = {(PAPER, "relu", 1.0)}
        if (model, bilip) in best:
            cands.add(best[(model, bilip)])
        for cfg, act, g in sorted(cands):
            v = (model, bilip) + cfg + (act, g)
            rs = dedup_seeds([r for r in clean[v] if r["ok"]])
            if rs:
                ratio = [r["rollout_vs_ic_noise"]["0.1"] / r["rollout_vs_ic_noise"]["0.0"] for r in rs]
                rows.append(f"| {NAME[model]} | {NAME[bilip]} | {vstr(v)} | " +
                            " | ".join(ms([r["rollout_vs_ic_noise"][l] for r in rs]) for l in IC_LEVELS) +
                            f" | {ms(ratio, 2)} |")
    return "\n".join(rows)


def plot_noise(series, path, best):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {0: "#1f77b4", 1: "#d62728"}
    for ax, model in zip(axes, ["hnn", "node"]):
        for v, (levels, ys) in sorted(series.items()):
            if v[0] != model:
                continue
            mu = np.array([y[0] for y in ys]); sd = np.array([y[1] for y in ys])
            x = np.array(levels) + 1e-3  # log axis, clean shown at 1e-3
            cfg, act, g = best[(v[0], 1)]
            ls = ":" if (v[1] == 0 and v[2:] == tuple(cfg) + (act, g)) else "-"
            ax.errorbar(x, mu, yerr=sd, color=colors[v[1]], ls=ls, marker="o", capsize=3,
                        label=f"{EMB[v[1]]}: {vstr(v)}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("relative Gaussian noise on training data (+1e-3; leftmost = clean)\n"
                      "solid: own selected config, dotted: inner product with the bilip config")
        ax.set_ylabel("test rollout error (geo. mean, T=150)")
        ax.set_title(NAME[model]); ax.legend(fontsize=7); ax.grid(alpha=.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    clean_runs = load("paper", "tune", "probe", "tune2", "tune3", "final", "control")
    clean = by(clean_runs)
    best = best_configs()
    noise_runs = load("noise") + [r for r in clean_runs if r["args"]["seed"] in (0, 1, 2)]
    noise_md, series = noise_section(noise_runs, best)
    control_md, control_series = control_section(load("noise", "control") + clean_runs, best)
    for v, sv in control_series.items():
        series.setdefault(v, sv)
    md = ["# Results", "",
          "Metric: geometric mean over 500 test trajectories x 150 steps of the state relative error (paper Table 1). "
          "mean ± std over seeds; 'diverged' = NaN / error ≥ 1. Selection uses the validation rollout only.", "",
          "## 1. Paper reproduction (3-100-5e-3, ReLU, seeds 0-2)", paper_section(clean), "",
          "## 2. Hyperparameter search (seed 0)", grid_section(load("tune", "probe", "tune2", "tune3")), "",
          "## 3. Paper config vs selected config (seeds 0-2)", final_section(clean, best), "",
          "## 4. Gaussian noise on the training data (val/test clean), test rollout", noise_md, "",
          "## 5. Same hyperparameters for both embeddings (the bilipschitz-selected ones)", control_md, "",
          "## 6. Perturbed test initial conditions (models trained on clean data)", ic_section(clean, best), ""]
    with open(os.path.join(OUT, "results.md"), "w") as f:
        f.write("\n".join(md))
    json.dump({f"{m}_{EMB[b]}": {"n_layers": c[0], "n_hidden": c[1], "lr": c[2], "act": a, "rbf_gamma_scale": g}
               for (m, b), (c, a, g) in best.items()}, open(os.path.join(OUT, "best_configs.json"), "w"), indent=1)
    if load("noise"):
        plot_noise(series, os.path.join(OUT, "noise_robustness.png"), best)
    print("\n".join(md))
