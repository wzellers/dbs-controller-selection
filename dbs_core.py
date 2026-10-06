import copy
import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

__version__ = "2.6.0"


@dataclass
class Config:
    hidden_dim: int = 64
    pd_value: float = 0.4
    tmax: int = 1500
    sim_time_per_step: int = 100
    dt: float = 0.01
    steps_per_episode: int = 15
    n_episodes: int = 1000
    batch_size: int = 64
    warmup_steps: int = 500
    checkpoint_every: int = 50
    seed: int = 123

    state_dim: int = 4
    action_dim: int = 2
    lr: float = 1e-4
    gamma: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    exploration_noise: float = 0.1

    freq_max: float = 185.0
    amp_max: float = 5000.0

    w_biomarker: float = 0.5
    w_energy: float = 0.2
    reward_scale: float = 0.3

    sgis_min: float = 300.0
    sgis_max: float = 1500.0
    rms_max: float = 150.0

    sd_bounds: tuple = (25.1250, 34.5420)
    activity_bounds: tuple = (634.5006, 1189.5806)
    mobility_bounds: tuple = (0.8066, 0.9101)
    complexity_bounds: tuple = (1.5052, 1.6830)

    run_dir: str = "runs"
    tag: str = ""

    def name(self):
        base = f"td3_h{self.hidden_dim}_pd{self.pd_value}"
        return f"{base}_{self.tag}" if self.tag else base

    def window_samples(self):
        return int(self.sim_time_per_step / self.dt)

    def bounds(self):
        return {"sd": self.sd_bounds, "activity": self.activity_bounds,
                "mobility": self.mobility_bounds, "complexity": self.complexity_bounds}


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh(),
        )

    def forward(self, state):
        return self.net(state)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q1_forward(self, state, action):
        return self.q1(torch.cat([state, action], dim=-1))


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=1_000_000):
        self.max_size = max_size
        self.ptr = self.size = 0
        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (torch.FloatTensor(self.states[idx]),
                torch.FloatTensor(self.actions[idx]),
                torch.FloatTensor(self.rewards[idx]),
                torch.FloatTensor(self.next_states[idx]),
                torch.FloatTensor(self.dones[idx]))

    def state_dict(self):
        return {"ptr": self.ptr, "size": self.size,
                "states": self.states[:self.size], "actions": self.actions[:self.size],
                "rewards": self.rewards[:self.size], "next_states": self.next_states[:self.size],
                "dones": self.dones[:self.size]}

    def load_state_dict(self, d):
        n = int(d["size"])
        self.ptr, self.size = int(d["ptr"]), n
        self.states[:n] = d["states"]
        self.actions[:n] = d["actions"]
        self.rewards[:n] = d["rewards"]
        self.next_states[:n] = d["next_states"]
        self.dones[:n] = d["dones"]


class TD3Agent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.total_updates = 0
        self.actor = Actor(cfg.state_dim, cfg.action_dim, cfg.hidden_dim)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic = Critic(cfg.state_dim, cfg.action_dim, cfg.hidden_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.lr)

    def select_action(self, state, add_noise=True):
        with torch.no_grad():
            action = self.actor(torch.FloatTensor(state).unsqueeze(0)).squeeze(0).numpy()
        if add_noise and self.cfg.exploration_noise > 0:
            action = action + np.random.normal(0, self.cfg.exploration_noise, size=action.shape)
        return np.clip(action, -1.0, 1.0)

    def train_step(self, buf, batch_size):
        if buf.size < batch_size:
            return None
        cfg = self.cfg
        states, actions, rewards, next_states, dones = buf.sample(batch_size)
        with torch.no_grad():
            noise = (torch.randn_like(actions) * cfg.policy_noise).clamp(-cfg.noise_clip, cfg.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(-1.0, 1.0)
            target_q = torch.min(*self.critic_target(next_states, next_actions))
            target = rewards + (1.0 - dones) * cfg.gamma * target_q
        q1, q2 = self.critic(states, actions)
        critic_loss = nn.functional.mse_loss(q1, target) + nn.functional.mse_loss(q2, target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.total_updates += 1
        if self.total_updates % cfg.policy_delay == 0:
            actor_loss = -self.critic.q1_forward(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            for src, tgt in [(self.actor, self.actor_target), (self.critic, self.critic_target)]:
                for sp, tp in zip(src.parameters(), tgt.parameters()):
                    tp.data.copy_(cfg.tau * sp.data + (1.0 - cfg.tau) * tp.data)
        return float(critic_loss.item())

    def param_counts(self):
        return (sum(p.numel() for p in self.actor.parameters()),
                sum(p.numel() for p in self.critic.parameters()))

    def state_dict(self):
        return {"actor": self.actor.state_dict(), "actor_target": self.actor_target.state_dict(),
                "actor_opt": self.actor_optimizer.state_dict(),
                "critic": self.critic.state_dict(), "critic_target": self.critic_target.state_dict(),
                "critic_opt": self.critic_optimizer.state_dict(),
                "total_updates": self.total_updates}

    def load_state_dict(self, d):
        self.actor.load_state_dict(d["actor"])
        self.actor_target.load_state_dict(d["actor_target"])
        self.actor_optimizer.load_state_dict(d["actor_opt"])
        self.critic.load_state_dict(d["critic"])
        self.critic_target.load_state_dict(d["critic_target"])
        self.critic_optimizer.load_state_dict(d["critic_opt"])
        self.total_updates = d["total_updates"]


def denormalize_action(action_norm, cfg):
    a = np.clip(action_norm, -1, 1)
    return float(cfg.freq_max * ((a[0] + 1) / 2)), float(cfg.amp_max * ((a[1] + 1) / 2))


def hjorth_raw(vgi, i_val, sim_time):
    window = vgi[:, i_val - sim_time:i_val:40].astype(np.float64)
    var_w = np.var(window, axis=1)
    var_d1 = np.var(np.diff(window), axis=1)
    var_d2 = np.var(np.diff(np.diff(window)), axis=1)
    mobilities = np.sqrt(var_d1 / var_w)
    sd = float(np.average(np.std(window, axis=1)))
    A = float(np.average(var_w))
    M = float(np.average(mobilities))
    C = float(np.average(np.sqrt(var_d2 / var_d1) / mobilities))
    return {"sd": sd, "activity": A, "mobility": M, "complexity": C,
            "window_shape": tuple(window.shape)}


def sgis_raw(sgis):
    return float(np.sum(np.abs(np.fft.fft(np.average(sgis.astype(np.float64), axis=0)))[1:20]))


class MatlabEnv:
    def __init__(self, cfg, mat_path="bgn_vars.mat"):
        import matlab.engine
        self.cfg = cfg
        self.mat_path = Path(mat_path)
        self.eng = matlab.engine.start_matlab()
        self.last = {}

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
                data = scipy.io.loadmat(self.mat_path)
                bad = [k for k in ("vgi", "dbs", "sgis")
                       if k in data and not np.all(np.isfinite(data[k]))]
                if bad:
                    err = f"non-finite values in {bad} (partial write?)"
                    time.sleep(delay)
                    continue
                return data
            except Exception as e:
                err = e
                time.sleep(delay)
        raise RuntimeError(f"Failed to load {self.mat_path}: {err}")

    def _unpack(self, data):
        cfg = self.cfg
        i_val = int(data["i"].flatten()[0])
        if i_val < cfg.window_samples():
            raise RuntimeError(
                f"i={i_val} is smaller than the {cfg.window_samples()}-sample window. "
                f"bgn_step is not advancing as expected — check sim_time_per_step and dt.")
        raw = hjorth_raw(data["vgi"], i_val, cfg.window_samples())
        raw["i"] = i_val
        raw["vgi_shape"] = tuple(data["vgi"].shape)
        raw["sgis_shape"] = tuple(data["sgis"].shape)
        raw["sgis"] = sgis_raw(data["sgis"])
        dbs = data["dbs"].flatten()[max(0, i_val - cfg.window_samples()):i_val].astype(np.float64)
        raw["rms"] = float(np.sqrt(np.mean(dbs ** 2))) if len(dbs) else 0.0
        if not np.isfinite(raw["rms"]):
            raise RuntimeError(f"non-finite rms at i={i_val}, dbs slice len {len(dbs)}")
        self.last = raw
        return raw

    def _state(self, raw):
        b = self.cfg.bounds()
        out = [(raw[k] - lo) / (hi - lo) for k, (lo, hi) in b.items()]
        return np.clip(np.array(out, dtype=np.float32), 0.0, 1.0)

    def _reward(self, raw):
        cfg = self.cfg
        r1 = float(np.clip((raw["sgis"] - cfg.sgis_min) / (cfg.sgis_max - cfg.sgis_min), 0, 1))
        r2 = float(np.clip(raw["rms"] / cfg.rms_max, 0, 1))
        base = -cfg.w_biomarker * r1 - cfg.w_energy * r2
        return base / cfg.reward_scale, base

    def reset(self):
        cfg = self.cfg
        self.eng.bgn_init(float(cfg.pd_value), float(cfg.tmax), nargout=0)
        if not self._wait():
            raise RuntimeError("bgn_vars.mat not ready after bgn_init")
        self.eng.bgn_step(0.0, 0.0, float(cfg.window_samples()), nargout=2)
        if not self._wait():
            raise RuntimeError("bgn_vars.mat not ready after initial bgn_step")
        return self._state(self._unpack(self._load()))

    def step(self, freq, amp):
        cfg = self.cfg
        self.eng.bgn_step(float(freq), float(amp), float(cfg.window_samples()), nargout=2)
        raw = self._unpack(self._load())
        td3_r, base_r = self._reward(raw)
        return self._state(raw), td3_r, base_r, raw["sgis"], raw["i"] >= cfg.tmax / cfg.dt

    def probe(self, freq, amp):
        self.step(freq, amp)
        return dict(self.last)

    def close(self):
        try:
            self.eng.quit()
        except Exception:
            pass


class MockEnv:
    def __init__(self, cfg, seed=0):
        self.cfg = cfg
        self.rng = np.random.RandomState(seed)
        self.last = {}

    def _raw(self, freq, amp):
        cfg = self.cfg
        drive = (freq / cfg.freq_max) * min(amp / 1500.0, 1.0)
        sgis = max(80.0, 740 * (0.6 + 0.4 * cfg.pd_value) * (1 - 0.5 * drive) + self.rng.randn() * 40)
        rms = amp * np.sqrt(max(freq, 0) / cfg.freq_max) / 26.0
        lo_sd, hi_sd = cfg.sd_bounds
        lo_a, hi_a = cfg.activity_bounds
        lo_m, hi_m = cfg.mobility_bounds
        lo_c, hi_c = cfg.complexity_bounds
        return {"sd": lo_sd + (hi_sd - lo_sd) * (0.3 + 0.5 * drive),
                "activity": lo_a + (hi_a - lo_a) * (0.3 + 0.5 * drive),
                "mobility": lo_m + (hi_m - lo_m) * (0.3 + 0.5 * drive),
                "complexity": hi_c - (hi_c - lo_c) * (0.3 + 0.5 * drive),
                "sgis": float(sgis), "rms": float(rms), "i": 0,
                "vgi_shape": (10, 1000), "sgis_shape": (10, 1000), "window_shape": (10, 25)}

    _state = MatlabEnv._state
    _reward = MatlabEnv._reward

    def reset(self):
        self.last = self._raw(0.0, 0.0)
        return self._state(self.last)

    def step(self, freq, amp):
        raw = self._raw(freq, amp)
        raw["sd"] += 0.05 * self.rng.randn()
        self.last = raw
        td3_r, base_r = self._reward(raw)
        return self._state(raw), td3_r, base_r, raw["sgis"], False

    def probe(self, freq, amp):
        self.step(freq, amp)
        return dict(self.last)

    def close(self):
        pass


class RunLogger:
    def __init__(self, cfg):
        self.dir = Path(cfg.run_dir) / cfg.name()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.csv = self.dir / "metrics.csv"
        if not self.csv.exists():
            self.csv.write_text("episode,base_reward,avg_freq,avg_amp,avg_sgis,buffer\n")
        payload = asdict(cfg)
        payload["_core_version"] = __version__
        (self.dir / "config.json").write_text(json.dumps(payload, indent=2, default=str))

    def log(self, ep, reward, freq, amp, sgis, buffer_size):
        with open(self.csv, "a") as f:
            f.write(f"{ep},{reward:.6f},{freq:.3f},{amp:.2f},{sgis:.2f},{buffer_size}\n")

    def checkpoint(self, ep, agent, buf=None):
        payload = {"episode": ep, "agent": agent.state_dict(), "version": __version__}
        torch.save(payload, self.dir / f"checkpoint_ep{ep:05d}.pt")
        torch.save(payload, self.dir / "latest.pt")
        if buf is not None:
            np.savez_compressed(self.dir / "buffer.npz", **buf.state_dict())

    def load_latest(self, agent, buf=None):
        path = self.dir / "latest.pt"
        if not path.exists():
            return 0
        payload = torch.load(path, weights_only=False)
        agent.load_state_dict(payload["agent"])
        bpath = self.dir / "buffer.npz"
        if buf is not None and bpath.exists():
            d = np.load(bpath)
            buf.load_state_dict({k: d[k] for k in d.files})
        return payload["episode"]


def collapsed(freqs, amps, window=50):
    if len(freqs) < window:
        return False
    return np.mean(freqs[-window:]) < 5.0 and np.mean(amps[-window:]) < 50.0


def train(cfg, env, resume=False, verbose=True):
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    agent = TD3Agent(cfg)
    buf = ReplayBuffer(cfg.state_dim, cfg.action_dim)
    logger = RunLogger(cfg)
    start_ep = logger.load_latest(agent, buf) if resume else 0

    if verbose:
        a, c = agent.param_counts()
        print(f"[{cfg.name()}] actor {a} | critic {c} | pd={cfg.pd_value} | "
              f"rms_max={cfg.rms_max} | start ep {start_ep}")

    total_steps = buf.size
    freq_hist, amp_hist = [], []

    for ep in range(start_ep + 1, cfg.n_episodes + 1):
        state = env.reset()
        ep_base, ep_f, ep_a, ep_s = 0.0, [], [], []
        for step in range(cfg.steps_per_episode):
            action = agent.select_action(state, add_noise=True)
            freq, amp = denormalize_action(action, cfg)
            next_state, td3_r, base_r, sgis, terminated = env.step(freq, amp)
            buf.add(state, action, td3_r, next_state, terminated)
            total_steps += 1
            if total_steps > cfg.warmup_steps:
                agent.train_step(buf, cfg.batch_size)
            state = next_state
            ep_base += base_r
            ep_f.append(freq)
            ep_a.append(amp)
            ep_s.append(sgis)
            if terminated:
                break

        n = len(ep_f)
        logger.log(ep, ep_base / n, np.mean(ep_f), np.mean(ep_a), np.mean(ep_s), buf.size)
        freq_hist.append(np.mean(ep_f))
        amp_hist.append(np.mean(ep_a))

        if verbose and ep % 10 == 0:
            print(f"  ep {ep:>4} | reward {ep_base / n:+.4f} | freq {np.mean(ep_f):6.1f} Hz | "
                  f"amp {np.mean(ep_a):6.0f} | buffer {buf.size}")

        if ep % cfg.checkpoint_every == 0:
            logger.checkpoint(ep, agent, buf)
            if collapsed(freq_hist, amp_hist):
                print(f"  !! ep {ep}: policy collapsed to no-stimulation "
                      f"(freq {np.mean(freq_hist[-50:]):.1f}, amp {np.mean(amp_hist[-50:]):.0f})")

    logger.checkpoint(cfg.n_episodes, agent, buf)
    return agent, logger


def evaluate(cfg, env, agent, n_episodes=25):
    rows = []
    for ep in range(n_episodes):
        state = env.reset()
        for step in range(cfg.steps_per_episode):
            action = agent.select_action(state, add_noise=False)
            freq, amp = denormalize_action(action, cfg)
            state, _, base_r, sgis, terminated = env.step(freq, amp)
            rows.append({"episode": ep, "step": step, "reward": base_r,
                         "freq": freq, "amp": amp, "sgis": sgis})
            if terminated:
                break
    return rows
