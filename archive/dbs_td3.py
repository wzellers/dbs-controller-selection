"""
TD3 deep-RL agent. Abandoned approach, kept for the record.

STATUS: superseded. Nothing in the live tree imports this file.

Why it was abandoned, since the reason is specific and worth keeping: the untrained
actor only ever proposed amplitudes between roughly 1600 and 3500 uA, so it never
sampled the low-amplitude region where the good settings turned out to be. It was
training on half the action space and lost to fixed open-loop stimulation.

The reward-model approach in dbs_reward_model.py replaced it, and removes that
failure mode by construction: actions are sampled uniformly by us, so the training
data covers the whole space.

Contents
    Actor, Critic    policy and value networks
    ReplayBuffer     experience storage
    TD3Agent         the agent
    RunLogger        checkpointing and metrics (TD3 only)
    collapsed        detects a policy that has collapsed to no stimulation
    train, evaluate  the training and evaluation loops

To run this you would need it back on the path alongside dbs_env.py.
"""

import copy
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dbs_env import Config, denormalize_action

__version__ = "2.6.0"


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
