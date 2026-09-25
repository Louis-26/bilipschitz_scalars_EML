"""
presentation figures from experiment/*/*/result.json -> output/figures/*.png

usage: python experiments/make_figures.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from make_tasks import PAPER, best_configs
from summarize import by, dedup_seeds, load, OUT, PAPER_TABLE1

FIG = os.path.join(OUT, "figures")
INNER, BILIP, REF = "#2a78d6", "#eb6834", "#8a8984"   # categorical slots 1-2 + neutral reference
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL = {0: INNER, 1: BILIP}
LAB = {0: "inner product $XX^\\top$", 1: "bilipschitz $(XX^\\top)^{1/2}$"}
NAME = {"hnn": "HNN", "node": "N-ODE"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight", "axes.titleweight": "bold",
})


def vals(rs, key="test"):
    return [r["final"]["test_Rollout"] if key == "test" else r["val_Rollout"] for r in rs if r["ok"]]


def bar(ax, x, v, color, label=None, w=0.36):
    m, s = np.mean(v), np.std(v)
    ax.bar(x, m, w, color=color, label=label, zorder=2, edgecolor="white", linewidth=2)
    ax.errorbar(x, m, yerr=s if len(v) > 1 else None, color=INK, capsize=4, lw=1.2, zorder=3)
    return m


def label_top(ax, x, m, txt=None):
    ax.annotate(txt or f"{m:.4f}", (x, m), xytext=(0, 5), textcoords="offset points", ha="center",
                fontsize=10, color=INK)


def fig_repro(clean):
    """paper setting: ours (inner) vs paper Table 1"""
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for i, model in enumerate(["hnn", "node"]):
        v = [r for r in clean[(model, 0) + PAPER + ("relu", 1.0)] if r["stage"] == "paper"]
        m = bar(ax, i - 0.2, vals(v), INNER, "ours (3 seeds)" if i == 0 else None)
        label_top(ax, i - 0.2, m + np.std(vals(v)), f"{m:.4f}")
        p, ps = PAPER_TABLE1[model]
        ax.bar(i + 0.2, p, 0.36, color=REF, label="paper Table 1" if i == 0 else None, zorder=2,
               edgecolor="white", linewidth=2)
        ax.errorbar(i + 0.2, p, yerr=ps, color=INK, capsize=4, lw=1.2, zorder=3)
        label_top(ax, i + 0.2, p + ps, f"{p:.3f}")
    ax.set_xticks([0, 1], [NAME["hnn"], NAME["node"]])
    ax.set_ylabel("test rollout error (lower = better)")
    ax.set_ylim(0, 0.0145)
    ax.legend(loc="upper left", ncol=2)
    fig.savefig(os.path.join(FIG, "repro.png"))
    plt.close(fig)


def fig_relu(runs_by_name):
    """training curves: why the bilipschitz HNN needs a smooth activation"""
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    spec = [("tune", "hnn_inner_L3_H100_lr0.01_noise0_s0", INNER, "-", "inner, ReLU", -8),
            ("tune", "hnn_bilip_L3_H100_lr0.01_noise0_s0", BILIP, ":", "bilip, ReLU", 0),
            ("tune2", "hnn_bilip_L3_H200_lr0.02_silu_noise0_s0", BILIP, "-", "bilip, SiLU", 8)]
    for stage, name, c, ls, lab, dy in spec:
        r = runs_by_name[(stage, name)]
        y = np.array(r["history"]["Train_MSE"]); x = np.array(r["history_step"])[:len(y)]
        ax.plot(x, y, color=c, ls=ls, lw=2, label=lab)
        ax.annotate(lab, (x[-1], y[-1]), xytext=(6, dy), textcoords="offset points", va="center",
                    fontsize=10, color=INK)
    ax.set_yscale("log")
    ax.set_xlabel("epoch"); ax.set_ylabel("training loss (MSE)")
    ax.set_xlim(0, 2600)
    ax.grid(axis="y", which="both", color=GRID)
    fig.savefig(os.path.join(FIG, "relu_vs_smooth.png"))
    plt.close(fig)


def fig_tuning(clean, best):
    """paper config vs selected config, 4 methods"""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
    for ax, model in zip(axes, ["hnn", "node"]):
        for j, bilip in enumerate([0, 1]):
            vp = [r for r in clean[(model, bilip) + PAPER + ("relu", 1.0)] if r["stage"] == "paper"]
            cfg, act, g = best[(model, bilip)]
            vb = dedup_seeds([r for r in clean[(model, bilip) + cfg + (act, g)]
                              if r["stage"] in ("tune", "probe", "tune2", "tune3", "final")])
            for k, (rs, alpha) in enumerate([(vp, 0.45), (vb, 1.0)]):
                x = j + (k - 0.5) * 0.38
                v = vals(rs)
                m = np.mean(v)
                ax.bar(x, m, 0.36, color=COL[bilip], alpha=alpha, zorder=2, edgecolor="white", linewidth=2)
                ax.errorbar(x, m, yerr=np.std(v), color=INK, capsize=3, lw=1, zorder=3)
                label_top(ax, x, m * (1 + (np.std(v) / m if m else 0)), f"{m:.4f}")
        ax.set_yscale("log")
        ax.set_xticks([-0.19, 0.19, 0.81, 1.19], ["paper\nconfig", "tuned", "paper\nconfig", "tuned"], fontsize=10)
        ax.set_title(NAME[model], color=INK)
        ax.text(0, -0.30, "inner", transform=ax.get_xaxis_transform(), ha="center", color=INNER, fontweight="bold")
        ax.text(1, -0.30, "bilipschitz", transform=ax.get_xaxis_transform(), ha="center", color=BILIP,
                fontweight="bold")
        ax.grid(axis="y", which="both", color=GRID)
    axes[0].set_ylabel("test rollout error (log)")
    axes[0].set_ylim(1e-3, 0.6)
    fig.savefig(os.path.join(FIG, "tuning.png"))
    plt.close(fig)


def curve(all_runs, v, levels):
    out = []
    for l in levels:
        rs = dedup_seeds([r for r in by(all_runs, l)[v] if r["ok"]])
        x = [r["final"]["test_Rollout"] for r in rs]
        out.append((np.mean(x), np.std(x)) if x else (np.nan, np.nan))
    return np.array(out)


def fig_noise(all_runs, best):
    """test rollout vs training noise, own tuned configs + same-config control"""
    levels = [0.0, 0.01, 0.03, 0.1]
    xs = np.array([0.003, 0.01, 0.03, 0.1])  # clean drawn at 0.003 on the log axis
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    for ax, model in zip(axes, ["hnn", "node"]):
        cfg_b = best[(model, 1)]
        lines = [((model, 0) + best[(model, 0)][0] + best[(model, 0)][1:], INNER, "-", "inner (own tuned)"),
                 ((model, 0) + cfg_b[0] + cfg_b[1:], INNER, ":", "inner (bilip's config)"),
                 ((model, 1) + cfg_b[0] + cfg_b[1:], BILIP, "-", "bilipschitz (tuned)")]
        for v, c, ls, lab in lines:
            y = curve(all_runs, v, levels)
            ax.errorbar(xs, y[:, 0], yerr=y[:, 1], color=c, ls=ls, lw=2, marker="o", ms=6, capsize=3, label=lab)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks(xs, ["clean", "0.01", "0.03", "0.1"])
        ax.minorticks_off()
        ax.set_xlabel("Gaussian noise on training data (σ, relative)")
        ax.set_title(NAME[model], color=INK)
        ax.grid(axis="both", which="major", color=GRID)
    axes[0].set_ylabel("test rollout error (log)")
    axes[1].legend(loc="upper left", fontsize=10)
    fig.savefig(os.path.join(FIG, "noise.png"))
    plt.close(fig)


def fig_control(all_runs, best):
    """relative error change bilip vs inner at identical hyperparameters"""
    levels = [0.0, 0.01, 0.03, 0.1]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    w = 0.36
    for j, model in enumerate(["hnn", "node"]):
        cfg, act, g = best[(model, 1)]
        yi = curve(all_runs, (model, 0) + cfg + (act, g), levels)
        yb = curve(all_runs, (model, 1) + cfg + (act, g), levels)
        d = 100 * (yb[:, 0] / yi[:, 0] - 1)
        x = np.arange(len(levels)) + (j - 0.5) * w
        ax.bar(x, d, w, color=BILIP if model == "node" else "#f2b79d", zorder=2, edgecolor="white", linewidth=2,
               label=f"{NAME[model]} ({cfg[0]}-{cfg[1]}-{cfg[2]:g}{', ' + act if act != 'relu' else ''})")
        for xi, di in zip(x, d):
            ax.annotate(f"{di:+.0f}%", (xi, di), xytext=(0, 4 if di >= 0 else -13), textcoords="offset points",
                        ha="center", fontsize=10, color=INK)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(range(len(levels)), ["clean", "σ = 0.01", "σ = 0.03", "σ = 0.1"])
    ax.set_ylabel("error of bilip vs inner (%)\n< 0: bilipschitz better")
    ax.set_ylim(-50, 35)
    ax.legend(loc="upper right", fontsize=10)
    fig.savefig(os.path.join(FIG, "control.png"))
    plt.close(fig)


def cfg_str(cfg, act):
    return f"{cfg[0]}-{cfg[1]}-{cfg[2]:g}" + (f", {act}" if act != "relu" else "")


def fig_clean(all_runs, best):
    """clean data: inner (own tuned), inner (bilip's config), bilip (tuned), with the config under each bar"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    for ax, model in zip(axes, ["hnn", "node"]):
        ci, ai, gi = best[(model, 0)]
        cb, ab, gb = best[(model, 1)]
        spec = [((model, 0) + ci + (ai, gi), INNER, 1.0, "inner\n(own tuned)", cfg_str(ci, ai)),
                ((model, 0) + cb + (ab, gb), INNER, 0.45, "inner\n(bilip's config)", cfg_str(cb, ab)),
                ((model, 1) + cb + (ab, gb), BILIP, 1.0, "bilipschitz\n(tuned)", cfg_str(cb, ab))]
        for x, (v, c, alpha, _, _) in enumerate(spec):
            m, s = curve(all_runs, v, [0.0])[0]
            ax.bar(x, m, 0.6, color=c, alpha=alpha, zorder=2, edgecolor="white", linewidth=2)
            ax.errorbar(x, m, yerr=s, color=INK, capsize=4, lw=1.2, zorder=3)
            label_top(ax, x, m + s, f"{m:.4f}")
        ax.set_xticks(range(3), [f"{t}\n[{cs}]" for *_, t, cs in spec], fontsize=10)
        ax.set_title(NAME[model], color=INK)
        ax.set_ylim(0, 0.0105)
    axes[0].set_ylabel("test rollout error (lower = better)")
    fig.suptitle("Clean training data (mean ± std, 3 seeds); [layers-width-lr]", fontsize=11, color=INK2, y=1.0)
    fig.savefig(os.path.join(FIG, "clean.png"))
    plt.close(fig)


def fig_ic_noise(clean, best):
    """models trained on clean data, Gaussian noise on the test initial conditions"""
    levels = ["0.0", "0.01", "0.03", "0.1"]
    xs = np.array([0.003, 0.01, 0.03, 0.1])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    for ax, model in zip(axes, ["hnn", "node"]):
        for bilip in [0, 1]:
            cfg, act, g = best[(model, bilip)]
            rs = dedup_seeds([r for r in clean[(model, bilip) + cfg + (act, g)]
                              if r["ok"] and r["stage"] in ("tune", "probe", "tune2", "tune3", "final")])
            y = np.array([[r["rollout_vs_ic_noise"][l] for r in rs] for l in levels])
            ax.errorbar(xs, y.mean(1), yerr=y.std(1), color=COL[bilip], lw=2, marker="o", ms=6, capsize=3,
                        label=f"{LAB[bilip]} [{cfg_str(cfg, act)}]")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks(xs, ["clean", "0.01", "0.03", "0.1"])
        ax.minorticks_off()
        ax.set_xlabel("Gaussian noise on test initial condition (σ_IC)")
        ax.set_title(NAME[model], color=INK)
        ax.grid(axis="both", which="major", color=GRID)
        ax.legend(loc="upper left", fontsize=9)
    axes[0].set_ylabel("test rollout error (log)")
    fig.savefig(os.path.join(FIG, "ic_noise.png"))
    plt.close(fig)


def fig_bound():
    """numerical check of the Balan-Dock bounds for theta(X) = (X X^T)^{1/2}, X in R^{4x3}"""
    rng = np.random.default_rng(0)

    def theta(X):
        U, S, _ = np.linalg.svd(X, full_matrices=False)
        return np.einsum("bik,bk,bjk->bij", U, S, U)

    def procrustes(X, Y):
        s = np.linalg.svd(np.einsum("bix,biy->bxy", X, Y), compute_uv=False).sum(-1)
        return np.sqrt(np.maximum((X ** 2).sum((1, 2)) + (Y ** 2).sum((1, 2)) - 2 * s, 0))

    n = 200_000
    X = rng.normal(size=(n, 4, 3)); Y = rng.normal(size=(n, 4, 3))
    Yc = X + 0.05 * rng.normal(size=(n, 4, 3))  # nearby pairs probe the local constants
    r1 = np.linalg.norm(theta(X) - theta(Y), axis=(1, 2)) / procrustes(X, Y)
    r2 = np.linalg.norm(theta(X) - theta(Yc), axis=(1, 2)) / procrustes(X, Yc)
    G = lambda Z: np.einsum("bix,bjx->bij", Z, Z)
    g1 = np.linalg.norm(G(X) - G(Y), axis=(1, 2)) / procrustes(X, Y)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bins = np.logspace(np.log10(0.8), np.log10(12), 120)
    ax.hist(g1, bins=bins, color=INNER, alpha=0.8, label="Gram $XX^\\top$ (random pairs)")
    ax.hist(np.r_[r1, r2], bins=bins, color=BILIP, alpha=0.9, label="$(XX^\\top)^{1/2}$ (random + nearby pairs)")
    for b in [1, np.sqrt(2)]:
        ax.axvline(b, color=INK, lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xlim(0.8, 12)
    ax.text(1.5, ax.get_ylim()[1] * 0.55, "theory: 1 ≤ ratio ≤ √2\n(Balan & Dock 2022)", fontsize=10, color=INK)
    ax.set_xticks([1, np.sqrt(2), 2, 5, 10], ["1", "√2", "2", "5", "10"])
    ax.minorticks_off()
    ax.set_xlabel("‖f(X) − f(Y)‖ / min$_U$‖X − YU‖   (X, Y ∈ ℝ$^{4×3}$)")
    ax.set_ylabel("count")
    ax.legend(loc="upper right", fontsize=9)
    fig.savefig(os.path.join(FIG, "bilip_bound.png"))
    plt.close(fig)
    return float(np.r_[r1, r2].min()), float(np.r_[r1, r2].max()), float(g1.min()), float(g1.max())


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    clean_runs = load("paper", "tune", "probe", "tune2", "tune3", "final", "control")
    clean = by(clean_runs)
    best = best_configs()
    all_runs = load("noise", "control") + clean_runs
    names = {(r["stage"], os.path.basename(r["args"]["out_dir"])): r for r in clean_runs}
    fig_repro(clean)
    fig_relu(names)
    fig_tuning(clean, best)
    fig_noise(all_runs, best)
    fig_control(all_runs, best)
    fig_clean(all_runs, best)
    fig_ic_noise(clean, best)
    print("bound check (min, max bilip ratio, min, max gram ratio):", fig_bound())
    print("\n".join(sorted(os.listdir(FIG))))
