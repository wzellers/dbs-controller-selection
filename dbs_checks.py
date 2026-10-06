import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

import dbs_core
from dbs_core import (Config, TD3Agent, ReplayBuffer, MockEnv, RunLogger,
                      denormalize_action, train, collapsed)

__version__ = "2.6.0"

CANDIDATE_ACTIONS = [(0, 0), (100, 1000), (130, 1000), (155, 1000),
                     (180, 1000), (155, 2500), (180, 5000)]


class Report:
    def __init__(self, title):
        self.title = title
        self.rows = []

    def add(self, name, ok, detail=""):
        self.rows.append((name, ok, detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

    def summary(self):
        bad = [n for n, ok, _ in self.rows if not ok]
        print(f"\n{self.title}: {len(self.rows) - len(bad)} passed, {len(bad)} failed")
        if bad:
            print("FAILED: " + ", ".join(bad))
        return not bad


def versions():
    import dbs_plots
    mods = {"dbs_core": dbs_core.__version__,
            "dbs_plots": dbs_plots.__version__,
            "dbs_checks": __version__}
    print("module versions:")
    for k, v in mods.items():
        print(f"  {k:<12} {v}")
    ok = len(set(mods.values())) == 1
    if not ok:
        print("\n  VERSION MISMATCH — one or more files on disk are stale.")
        print("  Re-download all four files, then restart the kernel.")
    else:
        print(f"\n  all modules at {mods['dbs_core']} — files are in sync")
    return ok


def offline(hidden_dim=64):
    r = Report(f"offline checks (hidden_dim={hidden_dim})")
    cfg = Config(hidden_dim=hidden_dim)

    agent = TD3Agent(cfg)
    s = torch.randn(8, cfg.state_dim)
    a = agent.actor(s)
    q1, q2 = agent.critic(s, a)
    r.add("actor output shape", tuple(a.shape) == (8, cfg.action_dim))
    r.add("actor bounded in [-1,1]", bool((a.abs() <= 1).all()))
    r.add("critic returns scalars", tuple(q1.shape) == (8, 1) and tuple(q2.shape) == (8, 1))
    r.add("twin critics differ", not torch.allclose(q1, q2))
    na, nc = agent.param_counts()
    print(f"         actor={na} critic={nc} params")

    r.add("action min -> (0,0)", denormalize_action(np.array([-1.0, -1.0]), cfg) == (0.0, 0.0))
    r.add("action max -> bounds",
          denormalize_action(np.array([1.0, 1.0]), cfg) == (cfg.freq_max, cfg.amp_max))

    buf = ReplayBuffer(cfg.state_dim, cfg.action_dim, max_size=10)
    for i in range(25):
        buf.add(np.full(cfg.state_dim, i), np.zeros(cfg.action_dim), -i,
                np.zeros(cfg.state_dim), False)
    r.add("buffer caps at max_size", buf.size == 10)
    r.add("buffer wraps", buf.ptr == 5)

    vgi = np.random.RandomState(0).randn(10, 20000) * 30
    raw = dbs_core.hjorth_raw(vgi, 15000, 4000)
    r.add("hjorth returns 4 finite values",
          all(np.isfinite(raw[k]) for k in ["sd", "activity", "mobility", "complexity"]),
          f"window {raw['window_shape']}")

    d = tempfile.mkdtemp()
    c = Config(**{**cfg.__dict__, "n_episodes": 2, "steps_per_episode": 5,
                  "warmup_steps": 10 ** 9, "checkpoint_every": 10 ** 9, "run_dir": d})
    agent2, logger = train(c, MockEnv(c), verbose=False)
    b2 = ReplayBuffer(c.state_dim, c.action_dim)
    logger.load_latest(agent2, b2)
    r.add("time-limit steps NOT marked done", float(b2.dones[:b2.size].sum()) == 0.0,
          f"{int(b2.dones[:b2.size].sum())} of {b2.size}")

    restored = TD3Agent(c)
    logger.checkpoint(42, agent2)
    ep = logger.load_latest(restored)
    same = all(torch.allclose(p, q) for p, q in
               zip(agent2.actor.parameters(), restored.actor.parameters()))
    r.add("checkpoint roundtrip", ep == 42 and same)
    shutil.rmtree(d, ignore_errors=True)

    d = tempfile.mkdtemp()
    c = Config(**{**cfg.__dict__, "n_episodes": 250, "steps_per_episode": 10,
                  "warmup_steps": 200, "checkpoint_every": 10 ** 9, "run_dir": d})
    _, logger = train(c, MockEnv(c), verbose=False)
    import csv
    rows = list(csv.DictReader(open(logger.dir / "metrics.csv")))
    rew = [float(x["base_reward"]) for x in rows]
    frq = [float(x["avg_freq"]) for x in rows]
    amp = [float(x["avg_amp"]) for x in rows]
    r.add("learns on mock env", np.mean(rew[-30:]) > np.mean(rew[:30]),
          f"{np.mean(rew[:30]):+.4f} -> {np.mean(rew[-30:]):+.4f}")
    r.add("no collapse on mock", not collapsed(frq, amp), f"final freq {np.mean(frq[-30:]):.1f} Hz")
    shutil.rmtree(d, ignore_errors=True)

    return r.summary()


def reward_landscape(cfg, probes):
    rows = []
    for (freq, amp), raw in probes.items():
        r1 = float(np.clip((raw["sgis"] - cfg.sgis_min) / (cfg.sgis_max - cfg.sgis_min), 0, 1))
        r2 = float(np.clip(raw["rms"] / cfg.rms_max, 0, 1))
        rows.append({"freq": freq, "amp": amp, "sgis": raw["sgis"], "rms": raw["rms"],
                     "r1": r1, "r2": r2,
                     "reward": -cfg.w_biomarker * r1 - cfg.w_energy * r2})
    return rows


def live(env, cfg=None, actions=CANDIDATE_ACTIONS, repeats=1,
         settle=3, measure=4, verbose=True):
    cfg = cfg or env.cfg
    r = Report("live environment checks")

    state = env.reset()
    raw0 = dict(env.last)
    r.add("reset returns correct state_dim", len(state) == cfg.state_dim)
    r.add("state values finite", bool(np.all(np.isfinite(state))))

    win = raw0.get("window_shape", (0, 0))
    expect = cfg.window_samples() // 40
    r.add("hjorth window has enough samples", win[1] > 4,
          f"window {win}, expected ~{expect} "
          f"(sim_time_per_step={cfg.sim_time_per_step} ms / dt={cfg.dt})")

    keys = ["sd", "activity", "mobility", "complexity", "sgis", "rms"]
    probes, samples = {}, {}
    for freq, amp in actions:
        reps = []
        for _ in range(repeats):
            env.reset()
            for _ in range(settle):
                env.step(freq, amp)
            for _ in range(measure):
                reps.append(env.probe(freq, amp))
        samples[(freq, amp)] = reps
        avg = {k: float(np.mean([r[k] for r in reps])) for k in keys}
        avg["window_shape"] = reps[0].get("window_shape")
        avg["sd_std"] = float(np.std([r["sd"] for r in reps]))
        avg["sgis_std"] = float(np.std([r["sgis"] for r in reps]))
        probes[(freq, amp)] = avg
    if verbose:
        print(f"\n  {repeats} episode(s) x {measure} steps per action "
              f"(first {settle} steps discarded as DBS turn-on transient)")

    if verbose:
        print()
        print(f"  {'action':<14} {'sgis':>9} {'rms':>8} {'r1':>6} {'r2':>6} {'reward':>9}")
        print("  " + "-" * 56)
    rows = reward_landscape(cfg, probes)
    for row in rows:
        if verbose:
            label = "no DBS" if row["freq"] == 0 else f"{row['freq']:.0f}Hz/{row['amp']:.0f}"
            sd = probes[(row["freq"], row["amp"])]["sgis_std"]
            print(f"  {label:<14} {row['sgis']:>9.1f} {row['rms']:>8.2f} "
                  f"{row['r1']:>6.3f} {row['r2']:>6.3f} {row['reward']:>9.4f}  "
                  f"(sgis sd {sd:.1f})")
    print()

    sgis_vals = [row["sgis"] for row in rows]
    r.add("sgis in plausible range", 50 < np.median(sgis_vals) < 3000,
          f"median {np.median(sgis_vals):.1f}, expected order 100-2000")

    b = cfg.bounds()
    for k, (lo, hi) in b.items():
        vals = [r[k] for reps in samples.values() for r in reps]
        inside = sum(1 for v in vals if lo <= v <= hi)
        r.add(f"feature '{k}' inside bounds", inside >= len(vals) * 0.5,
              f"{inside}/{len(vals)} in [{lo:.3f}, {hi:.3f}], "
              f"observed [{min(vals):.3f}, {max(vals):.3f}]")

    r1s = [row["r1"] for row in rows]
    r.add("biomarker term not saturated", 0 < np.mean(r1s) < 1,
          f"mean r1 = {np.mean(r1s):.3f}")

    rewards = [row["reward"] for row in rows]
    spread = max(rewards) - min(rewards)
    r.add("reward varies across actions", spread > 0.02, f"spread {spread:.4f}")

    no_dbs = next((row["reward"] for row in rows if row["freq"] == 0), None)
    if no_dbs is not None:
        best_stim = max(row["reward"] for row in rows if row["freq"] > 0)
        margin = best_stim - no_dbs
        r.add("stimulating beats doing nothing", margin > 0.02,
              f"margin {margin:+.4f} — below 0.02 risks no-stim collapse")

    ok = r.summary()
    if not ok:
        print("\nDo NOT start a long run until these pass. See suggest_bounds(samples).")
    return ok, samples, rows


def rms_range(samples):
    vals = _flat(samples, "rms")
    return min(vals), max(vals)


def _flat(samples, key):
    if samples and isinstance(next(iter(samples.values())), list):
        return [r[key] for reps in samples.values() for r in reps]
    return [p[key] for p in samples.values()]


def suggest_bounds(samples, headroom=0.20):
    n = sum(len(v) for v in samples.values()) if isinstance(
        next(iter(samples.values())), list) else len(samples)
    print(f"Suggested Config values from {n} observations "
          f"({headroom:.0%} headroom):\n")
    for k in ["sd", "activity", "mobility", "complexity"]:
        vals = _flat(samples, k)
        lo, hi = min(vals), max(vals)
        span = hi - lo
        pad = span * headroom if span > 0 else abs(hi) * 0.05 or 1e-6
        print(f"    {k}_bounds=({lo - pad:.4f}, {hi + pad:.4f}),")
    sg = _flat(samples, "sgis")
    rms = _flat(samples, "rms")
    print(f"    sgis_min={min(sg) * 0.90:.1f},")
    print(f"    sgis_max={max(sg) * 1.10:.1f},")
    print(f"\n  observed rms range [{min(rms):.2f}, {max(rms):.2f}]")
    print("  pick rms_max with tune_rms_max() — do not just use the max")


def baseline(env, cfg, actions=CANDIDATE_ACTIONS, n_episodes=5, settle=3):
    print(f"{'action':<14} {'all steps':>11} {'steady':>10} {'sgis':>8} {'rms':>8}")
    print("-" * 56)
    out = {}
    for freq, amp in actions:
        rows = []
        for _ in range(n_episodes):
            env.reset()
            for step in range(cfg.steps_per_episode):
                _, _, base_r, sgis, term = env.step(freq, amp)
                rows.append({"step": step, "reward": base_r, "sgis": sgis,
                             "rms": env.last["rms"]})
                if term:
                    break
        late = [x for x in rows if x["step"] >= settle]
        rec = {"all": float(np.mean([x["reward"] for x in rows])),
               "steady": float(np.mean([x["reward"] for x in late])),
               "sgis": float(np.mean([x["sgis"] for x in late])),
               "rms": float(np.mean([x["rms"] for x in late]))}
        out[(freq, amp)] = rec
        label = "no DBS" if freq == 0 else f"{freq:.0f}Hz/{amp:.0f}"
        print(f"{label:<14} {rec['all']:>+11.4f} {rec['steady']:>+10.4f} "
              f"{rec['sgis']:>8.0f} {rec['rms']:>8.1f}")
    best = max((k for k in out if k[0] > 0), key=lambda k: out[k]["all"])
    print(f"\nbest fixed action: {best[0]:.0f}Hz/{best[1]:.0f} "
          f"(all steps {out[best]['all']:+.4f}) — this is the o-DBS baseline to beat")
    return out


def tune_rms_max(cfg, probes, candidates=None):
    if candidates is None:
        top = max(p["rms"] for p in probes.values())
        candidates = [round(top * f, 1) for f in
                      (0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0)]
        print(f"observed max rms {top:.1f} -> candidates scaled from it\n")
    print(f"{'rms_max':>9} | {'best stim action':<16} | {'margin':>9} | {'winner amp':>10}")
    print("-" * 56)
    viable = []
    for rmax in candidates:
        c = Config(**{**cfg.__dict__, "rms_max": float(rmax)})
        rows = reward_landscape(c, probes)
        no_dbs = next((r["reward"] for r in rows if r["freq"] == 0), None)
        stim = [r for r in rows if r["freq"] > 0]
        if no_dbs is None or not stim:
            continue
        top = max(stim, key=lambda r: r["reward"])
        margin = top["reward"] - no_dbs
        label = f"{top['freq']:.0f}Hz/{top['amp']:.0f}"
        flag = ""
        if margin < 0.02:
            flag = "  <-- collapse risk"
        else:
            viable.append((rmax, margin, top["amp"]))
        print(f"{rmax:>9} | {label:<16} | {margin:>+9.4f} | {top['amp']:>10.0f}{flag}")
    if not viable:
        print("\n  no candidate keeps a workable margin — check the probe data")
        return None
    lowest_amp = min(v[2] for v in viable)
    best = max((v for v in viable if v[2] == lowest_amp), key=lambda v: v[1])
    print(f"\n  suggested rms_max = {best[0]} (margin {best[1]:+.4f}, "
          f"winner uses {best[2]:.0f} uA/cm2)")
    print("  chosen as the largest value whose best action still uses the "
          "lowest amplitude available")
    return best
