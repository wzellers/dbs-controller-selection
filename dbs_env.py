"""
Simulation environment and state features for the basal-ganglia-thalamic model.

STATUS: active. Everything in this file is used by current work.

Split out of the old dbs_core.py. The other half of that file -- the abandoned TD3
agent -- is now archive/dbs_td3.py and is not imported by anything live.

Contents
    Config              every tunable: simulation timing, reward weights, state bounds
    MatlabEnv           the real BGT model, driven through the MATLAB engine
    MockEnv             a stand-in needing no MATLAB, for exercising code paths
    hjorth_raw          the four state features: sd, activity, mobility, complexity
    sgis_raw            the STN-input biomarker this path's reward is built on
    denormalize_action  maps a [-1, 1] action onto real frequency and amplitude

Imported by
    dbs_reward_model.py, dbs_checks.py, dbs_reward_model.ipynb

KNOWN ISSUE -- how channels are combined.
    hjorth_raw computes each parameter per neuron and then averages the ten results.
    A real DBS lead records one combined field potential, so this is not the quantity
    the device would see. The two conventions are not interchangeable: averaging
    per-neuron statistics is blind to how synchronised the neurons are, while
    computing on the combined signal tracks synchrony directly. helpers.calculate_pb
    has the same issue; sgis_raw below does the opposite and combines first.
    See README, "Open questions".
"""

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

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


