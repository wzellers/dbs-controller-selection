import json
import os
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path

import numpy as np

__version__ = "3.1.0"

FREQ_VALUES = [55, 80, 105, 130, 155, 180]
CURR_VALUES = [0, 1000, 2000, 3000, 4000, 5000]


def build_arms():
    arms = [(FREQ_VALUES[0], 0)]
    arms += [(f, c) for f in FREQ_VALUES for c in CURR_VALUES if c > 0]
    return arms


ARMS = build_arms()


@dataclass
class T3PConfig:
    pd_value: float = 1.0
    tmax: int = 1000
    n_arms: int = len(ARMS)
    top_k: int = 25
    eps_start: float = 0.2
    eps_min: float = 0.0
    eps_decay: float = 0.025
    n_rounds: int = 75
    seed: int | None = 123
    w_pb: float = -0.7
    w_nostim: float = 0.1
    w_rms: float = -0.2
    pb_fn: str = "calculate_pb"
    pb_norm_fn: str = "normalize_pb"
    results_dir: str = "runs_t3p"
    tag: str = ""

    def name(self):
        base = f"t3p_pd{self.pd_value}_K{self.top_k}_eps{self.eps_start}"
        return f"{base}_{self.tag}" if self.tag else base


class MatlabBackend:
    def __init__(self, code_dir=None, parent_dir=None):
        import matlab.engine
        self.code_dir = Path(code_dir or Path.cwd())
        self.parent_dir = Path(parent_dir or self.code_dir / "matlab")
        self.mat_path = self.code_dir / "bgn_vars.mat"
        self.eng = matlab.engine.start_matlab()
        self.eng.cd(str(self.code_dir), nargout=0)
        self.eng.addpath(str(self.parent_dir), nargout=0)
        gating = self.parent_dir / "gating"
        if gating.exists():
            self.eng.addpath(str(gating), nargout=0)
        else:
            print(f"WARNING: {gating} not found — gating functions may be missing")
        print(f"MATLAB cwd {self.code_dir}\nMATLAB path + {self.parent_dir}, {gating}")

    def _wait(self, retries=10, delay=0.2):
        for _ in range(retries):
            if self.mat_path.exists():
                return True
            time.sleep(delay)
        return False

    def _load(self, retries=5, delay=0.2):
        import scipy.io
        err = None
        for _ in range(retries):
            try:
                return scipy.io.loadmat(self.mat_path)
            except Exception as e:
                err = e
                time.sleep(delay)
        raise RuntimeError(f"Failed to load {self.mat_path}: {err}")

    def simulate(self, freq, curr, cfg, seed=None):
        s = int(np.random.randint(9_999_999) if seed is None else seed)
        self.eng.bgn_init(float(cfg.pd_value), float(cfg.tmax), s, nargout=0)
        if not self._wait():
            raise RuntimeError("bgn_vars.mat not ready after bgn_init")
        safe_freq = float(freq) if curr > 0 else 130.0
        self.eng.bgn_step(safe_freq, float(curr), float(cfg.tmax) * 100, nargout=2)
        if not self._wait():
            raise RuntimeError("bgn_vars.mat not ready after bgn_step")
        data = self._load()
        for k in ("vgi", "dbs"):
            if k not in data:
                raise KeyError(f"Missing '{k}' in {self.mat_path}")
        return data["vgi"], data["dbs"]

    def close(self):
        try:
            self.eng.quit()
        except Exception:
            pass


class MockBackend:
    def __init__(self, seed=0):
        self.rng = np.random.RandomState(seed)

    def simulate(self, freq, curr, cfg, seed=None):
        pd = cfg.pd_value
        untreated = 33178 + (234149 - 33178) * (0.35 + 0.62 * pd)
        if curr == 0:
            pb = untreated
        else:
            f_eff = np.clip((freq - 40) / 145, 0, 1) ** 1.5
            a_eff = np.exp(-((curr - 1000) / 2200.0) ** 2)
            pb = untreated * (1 - 0.72 * f_eff * a_eff)
        pb = max(33178.0, pb + self.rng.randn() * 6000)
        n = int(cfg.tmax / 0.01) + 1
        dbs = np.zeros((1, n))
        if curr > 0:
            step = max(1, round((1000.0 / freq) / 0.01))
            width = int(0.3 / 0.01)
            for i in range(0, n, step):
                dbs[0, i:min(i + width, n)] = curr
        vgi = np.full((10, n), pb / 10.0)
        self._pb = pb
        return vgi, dbs


def compute_reward(vgi, dbs, freq, curr, cfg, helpers):
    if isinstance(getattr(helpers, "_mock_pb", None), float):
        pb = helpers._mock_pb
    else:
        pb = float(getattr(helpers, cfg.pb_fn)(vgi))
    r1 = getattr(helpers, cfg.pb_norm_fn)(pb)
    r2 = helpers.normalize_rms(helpers.compute_rms(dbs))
    r3 = helpers.normalize_consecutive_misses(
        helpers.count_max_consecutive_zeros(cfg.tmax, freq, curr))
    reward = cfg.w_pb * r1 + cfg.w_nostim * r3 + cfg.w_rms * r2
    return {"pb": pb, "r1": r1, "r2": r2, "r3": r3, "reward": float(reward)}


class T3PBandit:
    def __init__(self, cfg):
        self.cfg = cfg
        self.n = cfg.n_arms
        self.Q = np.zeros(self.n)
        self.counts = np.zeros(self.n, dtype=int)
        self.active = list(range(self.n))
        self.eps = cfg.eps_start
        self.t = 0
        self.pruned = False
        self.rng = np.random.default_rng(cfg.seed)

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.Q[arm] += (reward - self.Q[arm]) / self.counts[arm]

    def prune(self):
        order = np.argsort(-self.Q)
        self.active = sorted(int(a) for a in order[:self.cfg.top_k])
        self.pruned = True

    def select(self):
        self.t += 1
        self.eps = max(self.cfg.eps_min,
                       self.cfg.eps_start - self.cfg.eps_decay * self.t)
        if self.rng.random() < self.eps:
            return int(self.rng.choice(self.active)), "explore"
        best = max(self.active, key=lambda a: self.Q[a])
        return int(best), "exploit"

    def best_arm(self):
        return int(max(self.active, key=lambda a: self.Q[a]))


def sweep_all_arms(backend, cfg, helpers, repeats=1, verbose=True):
    rows = []
    if verbose:
        print(f"sweeping {len(ARMS)} arms at pd={cfg.pd_value}, {repeats} rep(s) each")
        print(f"biomarker: {cfg.pb_fn} / {cfg.pb_norm_fn}")
        print(f"{'idx':>4} {'arm':<14} {'pb':>10} {'r1':>7} {'r2':>7} {'r3':>7} {'reward':>9}")
        print("-" * 62)
    for i, (f, c) in enumerate(ARMS):
        reps = [compute_reward(*backend.simulate(f, c, cfg), f, c, cfg, helpers)
                for _ in range(repeats)]
        rec = {k: float(np.mean([r[k] for r in reps])) for k in reps[0]}
        rec.update({"idx": i, "freq": f, "curr": c,
                    "reward_sd": float(np.std([r["reward"] for r in reps]))})
        rows.append(rec)
        if verbose:
            label = "no DBS" if c == 0 else f"{f}Hz/{c}"
            print(f"{i:>4} {label:<14} {rec['pb']:>10.0f} {rec['r1']:>7.3f} "
                  f"{rec['r2']:>7.3f} {rec['r3']:>7.3f} {rec['reward']:>9.4f}")
    best = max(rows, key=lambda r: r["reward"])
    nodbs = rows[0]
    if verbose:
        label = "no DBS" if best["curr"] == 0 else f"{best['freq']}Hz/{best['curr']}"
        print(f"\nbest arm: {label} (reward {best['reward']:+.4f})")
        print(f"no-DBS arm: {nodbs['reward']:+.4f} "
              f"(margin {best['reward'] - nodbs['reward']:+.4f})")
        if best["curr"] == 0:
            print("  !! WARNING: no-DBS is the best arm — reward function needs review")
    return rows


def run_t3p(backend, cfg, helpers, verbose=True, save=True):
    bandit = T3PBandit(cfg)
    log = []
    t0 = time.time()

    if verbose:
        print(f"[{cfg.name()}] {cfg.n_arms} arms | K={cfg.top_k} | "
              f"eps {cfg.eps_start}->{cfg.eps_min} decay {cfg.eps_decay} | "
              f"{cfg.n_rounds} rounds after warm-up")
        print(f"  biomarker: {cfg.pb_fn} / {cfg.pb_norm_fn}")

    for arm in range(cfg.n_arms):
        f, c = ARMS[arm]
        rec = compute_reward(*backend.simulate(f, c, cfg), f, c, cfg, helpers)
        bandit.update(arm, rec["reward"])
        log.append({"phase": "warmup", "round": arm + 1, "arm": arm,
                    "freq": f, "curr": c, **rec})
        if verbose and (arm + 1) % 10 == 0:
            print(f"  warmup {arm+1}/{cfg.n_arms} | {(time.time()-t0)/60:.1f} min")

    bandit.prune()
    dropped = [i for i in range(cfg.n_arms) if i not in bandit.active]
    if verbose:
        print(f"  pruned {len(dropped)} arms, {len(bandit.active)} remain")
        print(f"  no-DBS arm {'PRUNED' if 0 in dropped else 'KEPT'}")

    for r in range(cfg.n_rounds):
        arm, mode = bandit.select()
        f, c = ARMS[arm]
        rec = compute_reward(*backend.simulate(f, c, cfg), f, c, cfg, helpers)
        bandit.update(arm, rec["reward"])
        log.append({"phase": mode, "round": cfg.n_arms + r + 1, "arm": arm,
                    "freq": f, "curr": c, "eps": bandit.eps, **rec})
        if verbose and (r + 1) % 10 == 0:
            b = bandit.best_arm()
            print(f"  round {cfg.n_arms+r+1:>3} | {mode:<7} {f}Hz/{c} "
                  f"r={rec['reward']:+.4f} | best so far {ARMS[b][0]}Hz/{ARMS[b][1]}")

    best = bandit.best_arm()
    bf, bc = ARMS[best]
    result = {"config": asdict(cfg), "best_arm": best, "best_freq": bf, "best_curr": bc,
              "best_Q": float(bandit.Q[best]), "active": bandit.active,
              "Q": bandit.Q.tolist(), "counts": bandit.counts.tolist(),
              "minutes": round((time.time() - t0) / 60, 1), "log": log}

    if verbose:
        print(f"\nconverged to {bf}Hz/{bc} (Q={bandit.Q[best]:+.4f}) "
              f"in {result['minutes']} min")
        if bc == 0:
            print("  !! WARNING: converged to no stimulation")

    if save:
        d = Path(cfg.results_dir)
        d.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        p = d / f"{cfg.name()}_{stamp}.json"
        p.write_text(json.dumps(result, indent=2))
        if verbose:
            print(f"  saved -> {p}")
    return result, bandit


def compare_severities(results, verbose=True):
    if verbose:
        print(f"\n{'pd':>6} {'best arm':<14} {'Q':>9} {'freq':>6} {'amp':>7}")
        print("-" * 48)
    rows = []
    for r in sorted(results, key=lambda x: x["config"]["pd_value"]):
        pd = r["config"]["pd_value"]
        label = "no DBS" if r["best_curr"] == 0 else f"{r['best_freq']}Hz/{r['best_curr']}"
        rows.append({"pd": pd, "freq": r["best_freq"], "amp": r["best_curr"],
                     "Q": r["best_Q"]})
        if verbose:
            print(f"{pd:>6} {label:<14} {r['best_Q']:>9.4f} "
                  f"{r['best_freq']:>6} {r['best_curr']:>7}")
    if verbose:
        freqs = [x["freq"] for x in rows]
        amps = [x["amp"] for x in rows]
        print(f"\nfrequency varies across severity: {len(set(freqs)) > 1}")
        print(f"amplitude varies across severity: {len(set(amps)) > 1}")
        if len(set(freqs)) == 1 and len(set(amps)) == 1:
            print("  -> all severities chose the same arm; a selector has nothing to do")
    return rows
