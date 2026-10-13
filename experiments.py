"""
Specialization and detection-under-stimulation experiments.

A T3P agent's output is a fixed (frequency, amplitude) pair, so "running a controller"
means passing that pair as the stimulation parameters. That makes all three experiments
here reuse existing machinery:

  train_agents      -> dbs_t3p.run_t3p, once per severity
  cross_matrix      -> dbs_t3p.compute_reward on constant-pd runs, paired by seed
  detect_under_stim -> pd_switch.run_seeds with freq/amp from an agent

Everything shares ONE MATLAB engine. bgn_vars.mat is a single shared state file, so two
engines corrupt each other. Pass backend.eng to the pd_switch functions.
"""

import json
from pathlib import Path

import numpy as np

import dbs_t3p as t3p
import pd_switch as ps

FINE_CURRENTS = [0, 100, 250, 500, 750, 1000, 2000, 3500, 5000]


def use_fine_grid(currents=None):
    t3p.CURR_VALUES = list(currents or FINE_CURRENTS)
    t3p.ARMS = t3p.build_arms()
    print(f"grid: {len(t3p.FREQ_VALUES)} freqs x {len(t3p.CURR_VALUES)-1} nonzero amps "
          f"+ no-DBS = {len(t3p.ARMS)} arms")
    print(f"  freq {t3p.FREQ_VALUES}")
    print(f"  curr {t3p.CURR_VALUES}")
    return t3p.ARMS


def grid_edges(freq, curr):
    edges = []
    if freq == min(t3p.FREQ_VALUES):
        edges.append(f"freq {freq} is the grid minimum")
    if freq == max(t3p.FREQ_VALUES):
        edges.append(f"freq {freq} is the grid maximum")
    nonzero = [c for c in t3p.CURR_VALUES if c > 0]
    if curr == 0:
        edges.append("no stimulation selected")
    elif curr == min(nonzero):
        edges.append(f"amp {curr} is the lowest nonzero value")
    elif curr == max(nonzero):
        edges.append(f"amp {curr} is the grid maximum")
    return edges


def estimate_minutes(n_sims, seconds_per_1000ms=3.9, tmax=1000):
    return n_sims * seconds_per_1000ms * (tmax / 1000.0) / 60.0


def train_agents(backend, helpers, pd_values=(0.3, 0.6, 1.0),
                 out_dir="runs_agents", verbose=True, **cfg_kw):
    Path(out_dir).mkdir(exist_ok=True)
    agents = {}
    for pd in pd_values:
        path = Path(out_dir) / f"agent_pd{pd}.json"
        if path.exists():
            agents[pd] = json.loads(path.read_text())
            if verbose:
                a = agents[pd]
                print(f"pd={pd}: cached {a['freq']}Hz/{a['curr']}uA")
            continue
        cfg = t3p.T3PConfig(pd_value=pd, n_arms=len(t3p.ARMS), **cfg_kw)
        result, _ = t3p.run_t3p(backend, cfg, helpers, verbose=verbose)
        agents[pd] = {"pd": pd, "freq": result["best_freq"], "curr": result["best_curr"],
                      "Q": result["best_Q"], "n_arms": len(t3p.ARMS),
                      "freq_values": list(t3p.FREQ_VALUES),
                      "curr_values": list(t3p.CURR_VALUES)}
        path.write_text(json.dumps(agents[pd], indent=2))
    return agents


def report_agents(agents):
    print(f"{'pd':>6}  {'setting':<16} {'Q':>9}  grid check")
    print("-" * 70)
    flagged = []
    for pd, a in sorted(agents.items()):
        label = "no DBS" if a["curr"] == 0 else f"{a['freq']}Hz / {a['curr']}uA"
        edges = grid_edges(a["freq"], a["curr"])
        print(f"{pd:>6}  {label:<16} {a['Q']:>9.4f}  "
              f"{'; '.join(edges) if edges else 'interior'}")
        if edges:
            flagged.append(pd)
    settings = {(a["freq"], a["curr"]) for a in agents.values()}
    print()
    if len(settings) == 1:
        print("!! every severity chose the same arm — a selector would have nothing to do")
    else:
        print(f"{len(settings)} distinct settings across {len(agents)} severities")
    if flagged:
        print(f"!! grid-edge selections at pd {flagged} — widen the grid before trusting these")
    return flagged


def cross_matrix(backend, helpers, agents, brain_pds=None, seeds=range(1, 11),
                 tmax=5000, out_dir="runs_matrix", verbose=True):
    brain_pds = sorted(agents) if brain_pds is None else list(brain_pds)
    seeds = [int(s) for s in seeds]
    Path(out_dir).mkdir(exist_ok=True)
    rows = []
    for brain in brain_pds:
        cfg = t3p.T3PConfig(pd_value=brain, tmax=tmax, n_arms=len(t3p.ARMS))
        for apd in sorted(agents):
            a = agents[apd]
            path = Path(out_dir) / f"agent{apd}_brain{brain}_t{tmax}.json"
            if path.exists():
                rows += json.loads(path.read_text())
                continue
            if verbose:
                print(f"agent pd={apd} ({a['freq']}Hz/{a['curr']}uA) on brain pd={brain} "
                      f"... ", end="", flush=True)
            got = []
            for s in seeds:
                rec = t3p.compute_reward(
                    *backend.simulate(a["freq"], a["curr"], cfg, seed=s),
                    a["freq"], a["curr"], cfg, helpers)
                got.append({"agent_pd": apd, "brain_pd": brain, "seed": s,
                            "freq": a["freq"], "curr": a["curr"], **rec})
            path.write_text(json.dumps(got, indent=2))
            rows += got
            if verbose:
                print(f"reward {np.mean([g['reward'] for g in got]):+.4f}")
    return rows


def _cell(rows, apd, bpd, key="reward"):
    sel = sorted((r for r in rows if r["agent_pd"] == apd and r["brain_pd"] == bpd),
                 key=lambda r: r["seed"])
    return np.array([r[key] for r in sel])


def report_matrix(rows, key="reward", higher_is_better=None):
    if higher_is_better is None:
        higher_is_better = key != "pb"
    pick = max if higher_is_better else min
    sign = 1.0 if higher_is_better else -1.0

    apds = sorted({r["agent_pd"] for r in rows})
    bpds = sorted({r["brain_pd"] for r in rows})
    n = len(_cell(rows, apds[0], bpds[0], key))

    direction = "higher is better" if higher_is_better else "lower is better"
    print(f"mean {key} +/- sem, n={n} paired seeds per cell  ({direction})\n")
    print(f"{'agent \\ brain':<16}" + "".join(f"{f'pd {b}':>20}" for b in bpds))
    print("-" * (16 + 20 * len(bpds)))
    for a in apds:
        line = f"{f'trained pd {a}':<16}"
        for b in bpds:
            v = _cell(rows, a, b, key)
            best = pick(apds, key=lambda x: _cell(rows, x, b, key).mean())
            mark = "*" if a == best else " "
            line += f"{v.mean():>+14.4f}{mark}{v.std(ddof=1)/np.sqrt(len(v)):>5.3f}"
        print(line)
    print(f"\n* = best agent for that brain ({direction})")

    print(f"\npaired comparison against the matched agent, per brain")
    print(f"{'brain':>8}  {'matched agent':<16} {'vs':<16} {'difference':>22}")
    print("-" * 70)
    for b in bpds:
        matched = b if b in apds else min(apds, key=lambda x: abs(x - b))
        base = _cell(rows, matched, b, key)
        for a in apds:
            if a == matched:
                continue
            d = sign * (base - _cell(rows, a, b, key))
            sem = d.std(ddof=1) / np.sqrt(len(d))
            verdict = "matched wins" if d.mean() - 2 * sem > 0 else (
                "other wins" if d.mean() + 2 * sem < 0 else "not separated")
            print(f"{b:>8}  {f'pd {matched}':<16} {f'pd {a}':<16} "
                  f"{d.mean():>+10.4f} +/- {sem:.4f}  {verdict}")
    return apds, bpds


def detect_under_stim(eng, helpers, agent, seeds=range(1, 31),
                      pd_start=0.3, pd_end=0.7, out_dir=None, **kw):
    tag = "nostim" if agent["curr"] == 0 else f"{agent['freq']}hz{agent['curr']}ua"
    out_dir = out_dir or f"runs_stim_pd{pd_start}to{pd_end}_{tag}"
    results, _ = ps.run_seeds(eng, seeds, helpers.calculate_pb, out_dir=out_dir,
                              pd_start=pd_start, pd_end=pd_end,
                              freq=float(agent["freq"]), amp=float(agent["curr"]), **kw)
    return results


def report_detection(label, results, thresh=7.0, switch_s=5.0):
    gaps, seps = ps.summarize(results, switch_s=switch_s)
    r = ps.evaluate(results, ps.detect_cp, switch_s=switch_s, thresh=thresh)
    print(f"{label:<28} gap {gaps.mean():+6.1f}%  separation {seps.mean():5.2f} sd "
          f"(worst {abs(seps).min():4.2f})  detected {r['ok']:2d}/{r['n']}  "
          f"latency {r['latency']:.2f}s")
    return gaps, seps, r


def plot_matrix(rows, key="reward", higher_is_better=None, out_png="specialization.png"):
    import matplotlib.pyplot as plt

    if higher_is_better is None:
        higher_is_better = key != "pb"
    apds = sorted({r["agent_pd"] for r in rows})
    bpds = sorted({r["brain_pd"] for r in rows})

    fig, ax = plt.subplots(figsize=(10, 5))
    for a in apds:
        m = np.array([_cell(rows, a, b, key).mean() for b in bpds])
        e = np.array([_cell(rows, a, b, key).std(ddof=1) /
                      np.sqrt(len(_cell(rows, a, b, key))) for b in bpds])
        line = ax.errorbar(bpds, m, yerr=e, marker="o", ms=5, capsize=3, lw=1.8,
                           label=f"trained at pd {a}")
        ax.axvline(a, ls=":", lw=1, color=line[0].get_color(), alpha=0.5)

    ax.set_xlabel("brain severity (pd)")
    ax.set_ylabel(f"mean {key}" + ("" if higher_is_better else "  (lower is better)"))
    ax.set_title("How far from its training severity does a controller stay competent?")
    ax.legend(title="dotted line = training severity")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return fig


class AdaptiveT3P(t3p.T3PBandit):
    """T3P that can track a change in the environment.

    Three changes, each addressing a reason the published bandit cannot re-adapt:
      alpha      constant step size instead of 1/counts, which has infinite memory
      eps_floor  epsilon never reaches zero, so other arms keep being sampled
      keep_all   skip pruning, since the arm that is best after the change may be
                 among the ones pruned before it
    """

    def __init__(self, cfg, alpha=0.1, eps_floor=0.05, keep_all=True):
        super().__init__(cfg)
        self.alpha = alpha
        self.eps_floor = eps_floor
        self.keep_all = keep_all

    def update(self, arm, reward):
        self.counts[arm] += 1
        if self.counts[arm] == 1:
            self.Q[arm] = reward
        else:
            self.Q[arm] += self.alpha * (reward - self.Q[arm])

    def select(self):
        self.t += 1
        self.eps = max(self.eps_floor,
                       self.cfg.eps_start - self.cfg.eps_decay * self.t)
        if self.rng.random() < self.eps:
            return int(self.rng.choice(self.active)), "explore"
        return int(max(self.active, key=lambda a: self.Q[a])), "exploit"

    def prune(self):
        if self.keep_all:
            self.pruned = True
            return
        super().prune()


def run_adaptive_switch(backend, helpers, pd_before=0.3, pd_after=1.0,
                        rounds_before=75, rounds_after=150, tmax=1000,
                        target_arm=None, oracle_reward=None, seed=123,
                        alpha=0.1, eps_floor=0.05, keep_all=True, verbose=True):
    """Let a live bandit converge at one severity, change the brain under it, and
    measure how long it takes to find new parameters."""
    from dataclasses import replace

    cfg = t3p.T3PConfig(pd_value=pd_before, tmax=tmax, n_arms=len(t3p.ARMS), seed=seed)
    bandit = AdaptiveT3P(cfg, alpha=alpha, eps_floor=eps_floor, keep_all=keep_all)
    log = []

    def pull(phase):
        arm, mode = (None, "warmup") if phase == "warmup" else bandit.select()
        return arm, mode

    for arm in range(cfg.n_arms):
        f, c = t3p.ARMS[arm]
        rec = t3p.compute_reward(*backend.simulate(f, c, cfg), f, c, cfg, helpers)
        bandit.update(arm, rec["reward"])
        log.append({"phase": "warmup", "pd": pd_before, "arm": arm,
                    "freq": f, "curr": c, "reward": rec["reward"]})
    bandit.prune()

    for _ in range(rounds_before):
        arm, mode = bandit.select()
        f, c = t3p.ARMS[arm]
        rec = t3p.compute_reward(*backend.simulate(f, c, cfg), f, c, cfg, helpers)
        bandit.update(arm, rec["reward"])
        log.append({"phase": f"before/{mode}", "pd": pd_before, "arm": arm,
                    "freq": f, "curr": c, "reward": rec["reward"]})

    arm_at_switch = bandit.best_arm()
    if verbose:
        f, c = t3p.ARMS[arm_at_switch]
        print(f"converged at pd={pd_before} to {f}Hz/{c}uA, switching brain to pd={pd_after}")

    cfg = replace(cfg, pd_value=pd_after)
    changed_at = reached_at = None
    for r in range(rounds_after):
        arm, mode = bandit.select()
        f, c = t3p.ARMS[arm]
        rec = t3p.compute_reward(*backend.simulate(f, c, cfg), f, c, cfg, helpers)
        bandit.update(arm, rec["reward"])
        best = bandit.best_arm()
        log.append({"phase": f"after/{mode}", "pd": pd_after, "arm": arm, "best": best,
                    "freq": f, "curr": c, "reward": rec["reward"]})
        if changed_at is None and best != arm_at_switch:
            changed_at = r + 1
        if reached_at is None and target_arm is not None and best == target_arm:
            reached_at = r + 1

    after = [row["reward"] for row in log if row["phase"].startswith("after")]
    out = {"pd_before": pd_before, "pd_after": pd_after, "seed": seed,
           "arm_at_switch": arm_at_switch,
           "setting_at_switch": t3p.ARMS[arm_at_switch],
           "final_arm": bandit.best_arm(),
           "final_setting": t3p.ARMS[bandit.best_arm()],
           "pulls_to_change": changed_at,
           "pulls_to_target": reached_at,
           "seconds_to_change": None if changed_at is None else changed_at * tmax / 1000.0,
           "mean_reward_after": float(np.mean(after)),
           "log": log}
    if oracle_reward is not None:
        out["regret_total"] = float(sum(oracle_reward - r for r in after))
        out["regret_per_pull"] = out["regret_total"] / len(after)

    if verbose:
        fa, ca = out["final_setting"]
        print(f"  selection first changed after {changed_at} pulls "
              f"({out['seconds_to_change']} s of brain time)"
              if changed_at else "  selection never changed")
        if target_arm is not None:
            print(f"  reached the pd={pd_after} optimum after "
                  f"{reached_at} pulls" if reached_at else
                  f"  never reached the pd={pd_after} optimum in {rounds_after} pulls")
        print(f"  ended on {fa}Hz/{ca}uA")
        if oracle_reward is not None:
            print(f"  reward lost vs always playing the right arm: "
                  f"{out['regret_total']:+.3f} total, {out['regret_per_pull']:+.4f}/pull")
    return out


def report_adaptive(results, detector_latency_s=1.25):
    changed = [r["pulls_to_change"] for r in results if r["pulls_to_change"]]
    secs = [r["seconds_to_change"] for r in results if r["seconds_to_change"]]
    reached = [r["pulls_to_target"] for r in results if r["pulls_to_target"]]
    print(f"n = {len(results)} seeds")
    print(f"  selection changed in {len(changed)}/{len(results)} runs")
    if secs:
        print(f"  pulls to change      {np.mean(changed):6.1f} +/- "
              f"{np.std(changed, ddof=1)/np.sqrt(len(changed)):.1f}")
        ratio = np.mean(secs) / detector_latency_s
        rel = f"{ratio:.1f}x slower" if ratio >= 1 else f"{1/ratio:.1f}x faster"
        print(f"  brain time to change {np.mean(secs):6.1f} s  "
              f"vs {detector_latency_s:.2f} s for the P_beta detector ({rel})")
    if reached:
        secs_r = [r["pulls_to_target"] * (r["seconds_to_change"] / r["pulls_to_change"])
                  for r in results if r["pulls_to_target"] and r["pulls_to_change"]]
        extra = f" ({np.mean(secs_r):.0f} s of brain time)" if secs_r else ""
        print(f"  reached the new optimum in {len(reached)}/{len(results)} runs, "
              f"after {np.mean(reached):.1f} pulls{extra}")
    else:
        print(f"  never reached the new optimum in any run")
    if any("regret_total" in r for r in results):
        reg = [r["regret_total"] for r in results if "regret_total" in r]
        print(f"  reward lost while re-exploring: {np.mean(reg):+.3f} "
              f"+/- {np.std(reg, ddof=1)/np.sqrt(len(reg)):.3f}")


def arm_index(freq, curr):
    for i, (f, c) in enumerate(t3p.ARMS):
        if f == freq and c == curr:
            return i
    raise ValueError(f"{freq}Hz/{curr}uA is not in the current arm grid")


def plot_adaptive(results, oracle_reward=None, out_png="adaptive_reexploration.png"):
    import matplotlib.pyplot as plt

    traces = []
    for r in results:
        after = [row["reward"] for row in r["log"] if row["phase"].startswith("after")]
        before = [row["reward"] for row in r["log"] if row["phase"].startswith("before")]
        traces.append((before[-40:], after))
    nb = min(len(b) for b, _ in traces)
    na = min(len(a) for _, a in traces)
    pre = np.vstack([b[-nb:] for b, _ in traces])
    post = np.vstack([a[:na] for _, a in traces])
    x = np.arange(-nb, na)
    mean = np.concatenate([pre.mean(0), post.mean(0)])
    k = 9
    smooth = np.convolve(mean, np.ones(k) / k, mode="same")

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(x, mean, lw=0.8, alpha=0.35, color="tab:purple")
    ax.plot(x, smooth, lw=2.2, color="tab:purple", label=f"mean of {len(results)} seeds")
    ax.axvline(0, ls="--", lw=1.5, color="tab:red")
    ax.text(0, ax.get_ylim()[0], "  brain changes", color="tab:red", va="bottom", fontsize=10)
    if oracle_reward is not None:
        ax.axhline(oracle_reward, ls=":", lw=1.4, color="tab:green")
        ax.text(x[-1], oracle_reward, "right controller, applied instantly  ",
                ha="right", va="bottom", color="tab:green", fontsize=9)
    changed = [r["pulls_to_change"] for r in results if r["pulls_to_change"]]
    reached = [r["pulls_to_target"] for r in results if r["pulls_to_target"]]
    if changed:
        ax.axvline(np.mean(changed), ls="-.", lw=1.2, color="tab:orange")
        ax.text(np.mean(changed), ax.get_ylim()[1], " agent drops its old setting",
                color="tab:orange", va="top", fontsize=9)
    if reached:
        ax.axvline(np.mean(reached), ls="-.", lw=1.2, color="tab:blue")
        ax.text(np.mean(reached), ax.get_ylim()[1], " agent finds the new optimum",
                color="tab:blue", va="top", fontsize=9)
    ax.set_xlabel("pulls relative to the change in brain state")
    ax.set_ylabel("reward per pull")
    ax.set_title("What it costs to let the agent re-learn instead of switching controller")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return fig
