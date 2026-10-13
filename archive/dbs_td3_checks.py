"""
Offline checks for the retired TD3 agent.

STATUS: superseded, alongside archive/dbs_td3.py. Nothing live imports this.

These were part of dbs_checks.offline() while TD3 was the active approach. They
verify network shapes, replay-buffer wrap-around, checkpoint round-trips, and that
the agent can learn on the mock environment at all.

Needs archive/dbs_td3.py and dbs_env.py importable to run.
"""

import csv
import shutil
import tempfile

import numpy as np
import torch

from dbs_checks import Report
from dbs_env import Config, MockEnv, denormalize_action
from dbs_td3 import TD3Agent, ReplayBuffer, train, collapsed


def offline_td3(hidden_dim=64):
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

    buf = ReplayBuffer(cfg.state_dim, cfg.action_dim, max_size=10)
    for i in range(25):
        buf.add(np.full(cfg.state_dim, i), np.zeros(cfg.action_dim), -i,
                np.zeros(cfg.state_dim), False)
    r.add("buffer caps at max_size", buf.size == 10)
    r.add("buffer wraps", buf.ptr == 5)

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
