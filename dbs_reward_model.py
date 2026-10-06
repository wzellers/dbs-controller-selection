import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dbs_core import Config, denormalize_action

__version__ = "2.6.0"

CANDIDATE_GRID = [(f, a) for f in (55, 80, 105, 130, 155, 180)
                  for a in (0, 1000, 2000, 3000, 4000, 5000)]


def collect(env, cfg, n_samples=400, hold=4, seed=0, out_dir="datasets", verbose=True):
    rng = np.random.RandomState(seed)
    rows = []
    steps_per_sample = hold
    per_episode = max(1, cfg.steps_per_episode // hold)
    n_episodes = int(np.ceil(n_samples / per_episode))
    if verbose:
        total = n_episodes * per_episode * hold
        print(f"collecting {n_samples} samples at pd={cfg.pd_value} "
              f"(hold={hold}, {n_episodes} episodes, ~{total} MATLAB steps)")
    t0 = time.time()

    for ep in range(n_episodes):
        state = env.reset()
        for _ in range(per_episode):
            if len(rows) >= n_samples:
                break
            action_norm = rng.uniform(-1, 1, cfg.action_dim)
            freq, amp = denormalize_action(action_norm, cfg)
            state_before = state.copy()
            terminated = False
            for h in range(hold):
                state, _, _, _, terminated = env.step(freq, amp)
                if terminated:
                    break
            raw = env.last
            rows.append({"pd": cfg.pd_value,
                         **{f"s{i}": float(v) for i, v in enumerate(state_before)},
                         "a_freq_norm": float(action_norm[0]),
                         "a_amp_norm": float(action_norm[1]),
                         "freq": freq, "amp": amp,
                         "sgis": float(raw["sgis"]), "rms": float(raw["rms"])})
            if terminated:
                break
        if verbose and (ep + 1) % 5 == 0:
            print(f"  {len(rows)}/{n_samples} samples | {(time.time()-t0)/60:.1f} min")

    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"pd{cfg.pd_value}_n{len(rows)}.json"
    path.write_text(json.dumps(rows))
    if verbose:
        print(f"  saved {len(rows)} samples to {path} in {(time.time()-t0)/60:.1f} min")
    return rows


def to_tensors(rows, cfg, state_dim=None):
    state_dim = state_dim or cfg.state_dim
    X, y = [], []
    for r in rows:
        s = [r[f"s{i}"] for i in range(state_dim)]
        X.append(s + [r["a_freq_norm"], r["a_amp_norm"]])
        r1 = np.clip((r["sgis"] - cfg.sgis_min) / (cfg.sgis_max - cfg.sgis_min), 0, 1)
        r2 = np.clip(r["rms"] / cfg.rms_max, 0, 1)
        y.append(-cfg.w_biomarker * r1 - cfg.w_energy * r2)
    return torch.FloatTensor(X), torch.FloatTensor(y).unsqueeze(1)


class RewardModel(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def fit(rows, cfg, hidden_dim=64, epochs=3000, lr=1e-3, val_frac=0.2, seed=0, verbose=True):
    torch.manual_seed(seed)
    X, y = to_tensors(rows, cfg)
    n = len(X)
    idx = np.random.RandomState(seed).permutation(n)
    n_val = int(n * val_frac)
    va, tr = idx[:n_val], idx[n_val:]
    Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]

    model = RewardModel(X.shape[1], hidden_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, patience = float("inf"), None, 0
    for ep in range(epochs):
        opt.zero_grad()
        nn.functional.mse_loss(model(Xtr), ytr).backward()
        opt.step()
        if ep % 50 == 0:
            with torch.no_grad():
                v = float(nn.functional.mse_loss(model(Xva), yva))
            if v < best - 1e-6:
                best, best_state, patience = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
            else:
                patience += 1
                if patience >= 10:
                    break
    if best_state:
        model.load_state_dict(best_state)
    with torch.no_grad():
        tr_mse = float(nn.functional.mse_loss(model(Xtr), ytr))
        va_pred = model(Xva)
        va_mse = float(nn.functional.mse_loss(va_pred, yva))
        ss_res = float(((yva - va_pred) ** 2).sum())
        ss_tot = float(((yva - yva.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    if verbose:
        print(f"  hidden={hidden_dim:<3} params={model.n_params():<6} "
              f"train MSE {tr_mse:.5f} | val MSE {va_mse:.5f} | val R2 {r2:.3f}")
    return model, {"train_mse": tr_mse, "val_mse": va_mse, "val_r2": r2,
                   "hidden_dim": hidden_dim, "n_params": model.n_params(),
                   "n_train": len(tr), "n_val": len(va)}


def best_action(model, state, cfg, grid=CANDIDATE_GRID):
    rows = []
    for freq, amp in grid:
        fn = 2 * freq / cfg.freq_max - 1
        an = 2 * amp / cfg.amp_max - 1
        rows.append(list(state) + [fn, an])
    with torch.no_grad():
        preds = model(torch.FloatTensor(rows)).squeeze(1).numpy()
    i = int(np.argmax(preds))
    return grid[i], float(preds[i]), preds


def cross_evaluate(models, datasets, cfg, verbose=True):
    names = sorted(datasets)
    out = {}
    for mname in sorted(models):
        for dname in names:
            X, y = to_tensors(datasets[dname], cfg)
            with torch.no_grad():
                pred = models[mname](X)
                mse = float(nn.functional.mse_loss(pred, y))
                ss_res = float(((y - pred) ** 2).sum())
                ss_tot = float(((y - y.mean()) ** 2).sum())
            out[(mname, dname)] = {"mse": mse,
                                   "r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")}
    if verbose:
        print(f"\n{'model \\ data':<16}" + "".join(f"{d:>14}" for d in names))
        print("-" * (16 + 14 * len(names)))
        for mname in sorted(models):
            row = "".join(f"{out[(mname, d)]['r2']:>14.3f}" for d in names)
            print(f"{mname:<16}{row}")
        print("\n(values are val R2 — how well each model predicts each severity's rewards)")
    return out


def action_agreement(models, cfg, states, grid=CANDIDATE_GRID, verbose=True):
    picks = {}
    for mname, model in sorted(models.items()):
        counts = {}
        for s in states:
            act, _, _ = best_action(model, s, cfg, grid)
            counts[act] = counts.get(act, 0) + 1
        top = max(counts, key=counts.get)
        picks[mname] = {"modal_action": top, "share": counts[top] / len(states),
                        "n_distinct": len(counts)}
        if verbose:
            print(f"{mname:<16} picks {top[0]:>3.0f}Hz/{top[1]:>4.0f}uA "
                  f"in {100 * counts[top] / len(states):.0f}% of states "
                  f"({len(counts)} distinct actions)")
    return picks
