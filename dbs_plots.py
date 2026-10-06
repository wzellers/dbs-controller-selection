import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

__version__ = "2.6.0"


def _read_metrics(run_dir):
    rows = list(csv.DictReader(open(Path(run_dir) / "metrics.csv")))
    return {k: np.array([float(r[k]) for r in rows])
            for k in ["episode", "base_reward", "avg_freq", "avg_amp", "avg_sgis"]}


def _smooth(x, w=25):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def plot_training(run_dir, save=True):
    m = _read_metrics(run_dir)
    ep = m["episode"]
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))

    ax[0, 0].plot(ep, m["base_reward"], alpha=0.25, color="tab:blue")
    s = _smooth(m["base_reward"])
    ax[0, 0].plot(ep[len(ep) - len(s):], s, color="tab:blue", lw=2)
    ax[0, 0].set_title("Base reward per episode")
    ax[0, 0].set_xlabel("episode")

    ax[0, 1].plot(ep, m["avg_freq"], color="tab:orange", alpha=0.4)
    s = _smooth(m["avg_freq"])
    ax[0, 1].plot(ep[len(ep) - len(s):], s, color="tab:orange", lw=2)
    ax[0, 1].axhline(5, ls="--", color="red", lw=1)
    ax[0, 1].set_title("Avg stimulation frequency (Hz)\nred line = collapse threshold")
    ax[0, 1].set_xlabel("episode")

    ax[1, 0].plot(ep, m["avg_amp"], color="tab:green", alpha=0.4)
    s = _smooth(m["avg_amp"])
    ax[1, 0].plot(ep[len(ep) - len(s):], s, color="tab:green", lw=2)
    ax[1, 0].axhline(50, ls="--", color="red", lw=1)
    ax[1, 0].set_title("Avg stimulation amplitude (uA/cm2)")
    ax[1, 0].set_xlabel("episode")

    ax[1, 1].plot(ep, m["avg_sgis"], color="tab:purple", alpha=0.4)
    s = _smooth(m["avg_sgis"])
    ax[1, 1].plot(ep[len(ep) - len(s):], s, color="tab:purple", lw=2)
    ax[1, 1].set_title("Avg S_GPi spectral power (biomarker)")
    ax[1, 1].set_xlabel("episode")

    for a in ax.flat:
        a.grid(alpha=0.3)
    fig.suptitle(Path(run_dir).name)
    fig.tight_layout()
    if save:
        fig.savefig(Path(run_dir) / "training.png", dpi=130)
    return fig


def plot_eval(run_dir, rows, save=True):
    steps = sorted({r["step"] for r in rows})
    by_step = {k: [np.mean([r[k] for r in rows if r["step"] == s]) for s in steps]
               for k in ["reward", "freq", "amp", "sgis"]}

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    ax[0].plot(steps, by_step["reward"], marker="o", color="tab:blue")
    ax[0].set_title("Reward by step within episode")
    ax[0].set_xlabel("step")

    ax[1].plot(steps, by_step["freq"], marker="o", color="tab:orange", label="freq (Hz)")
    ax[1].set_xlabel("step")
    ax[1].set_ylabel("freq (Hz)")
    twin = ax[1].twinx()
    twin.plot(steps, by_step["amp"], marker="s", color="tab:green", label="amp")
    twin.set_ylabel("amp (uA/cm2)")
    ax[1].set_title("Policy trajectory within episode")

    sc = ax[2].scatter([r["freq"] for r in rows], [r["amp"] for r in rows],
                       c=[r["reward"] for r in rows], cmap="viridis", s=14)
    fig.colorbar(sc, ax=ax[2], label="reward")
    ax[2].set_xlabel("freq (Hz)")
    ax[2].set_ylabel("amp (uA/cm2)")
    ax[2].set_title("Visited action space")

    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle(f"{Path(run_dir).name} — evaluation")
    fig.tight_layout()
    if save:
        fig.savefig(Path(run_dir) / "eval.png", dpi=130)
    return fig


def compare_runs(run_dirs, save_to=None):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for d in run_dirs:
        m = _read_metrics(d)
        label = Path(d).name
        for i, k in enumerate(["base_reward", "avg_freq", "avg_amp"]):
            s = _smooth(m[k])
            ax[i].plot(m["episode"][len(m["episode"]) - len(s):], s, lw=2, label=label)
    for i, t in enumerate(["Base reward", "Avg freq (Hz)", "Avg amp (uA/cm2)"]):
        ax[i].set_title(t)
        ax[i].set_xlabel("episode")
        ax[i].grid(alpha=0.3)
        ax[i].legend(fontsize=8)
    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=130)
    return fig
