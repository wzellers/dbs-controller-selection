"""
P_beta across a mid-simulation pd switch.

CONSTRAINT: calculate_pb sums FFT bin indices 13:35, not frequencies. A bin maps
to k * Fs / N. With dt = 0.01 ms (Fs = 100 kHz) a 1000 ms window gives
N = 100,000 ~= Fs, so bin 13 lands on 13.0 Hz and bin 35 on 35.0 Hz. That is the
only window length for which this function measures the beta band. At 500 ms it
measures 26-70 Hz, at 100 ms it measures 130-350 Hz. Do not change WINDOW_MS
without switching to a frequency-indexed P_beta.

Requires set_pd.m on the MATLAB path (workspace/matlab/).
"""

import json
import time
from pathlib import Path

import numpy as np
import scipy.io

DT_MS = 0.01
FS_HZ = 100_000.0
WINDOW_MS = 1000
STEP_MS = 250
WINDOW_SAMPLES = int(WINDOW_MS / DT_MS)
STEP_SAMPLES = int(STEP_MS / DT_MS)
HALF_WINDOW_S = WINDOW_MS / 2000.0


def start_engine(code_dir=".", matlab_dir="matlab"):
    import matlab.engine
    code_dir = Path(code_dir).resolve()
    eng = matlab.engine.start_matlab()
    eng.cd(str(code_dir), nargout=0)
    eng.addpath(str(code_dir / matlab_dir), nargout=0)
    gating = code_dir / "gating"
    if gating.exists():
        eng.addpath(str(gating), nargout=0)
    return eng


def run_switch(eng, pd_start=0.3, pd_end=0.7, warmup_ms=2_000, hold_ms=5_000,
               freq=130.0, amp=0.0, seed=123, mat_path="bgn_vars.mat", verbose=True):
    t0 = time.time()
    total_ms = warmup_ms + 2 * hold_ms
    if verbose:
        print(f"bgn_init(pd={pd_start}, tmax={total_ms})")
    eng.bgn_init(float(pd_start), float(total_ms), int(seed), nargout=0)

    if verbose:
        print(f"  warmup + segment 1: pd={pd_start} for {warmup_ms + hold_ms} ms")
    eng.bgn_step(float(freq), float(amp), float((warmup_ms + hold_ms) / DT_MS), nargout=2)

    if verbose:
        print(f"  switching pd -> {pd_end}")
    eng.set_pd(float(pd_end), nargout=0)

    if verbose:
        print(f"  segment 2: pd={pd_end} for {hold_ms} ms")
    eng.bgn_step(float(freq), float(amp), float(hold_ms / DT_MS), nargout=2)

    data = scipy.io.loadmat(mat_path)
    vgi = np.asarray(data["vgi"], dtype=np.float64)[:, int(warmup_ms / DT_MS):]
    if verbose:
        print(f"  done in {time.time() - t0:.1f}s | vgi {vgi.shape}")
    return vgi


def sliding_pb(vgi, pb_fn, window_samples=WINDOW_SAMPLES, step_samples=STEP_SAMPLES):
    n = vgi.shape[1]
    times, values = [], []
    for s in range(0, n - window_samples + 1, step_samples):
        values.append(float(pb_fn(vgi[:, s:s + window_samples])))
        times.append((s + window_samples / 2) * DT_MS / 1000.0)
    return np.array(times), np.array(values)


def _settings(pb_fn, zoom_w, kw):
    return json.dumps({
        "pd_start": kw.get("pd_start", 0.3),
        "pd_end": kw.get("pd_end", 0.7),
        "warmup_ms": kw.get("warmup_ms", 2_000),
        "hold_ms": kw.get("hold_ms", 5_000),
        "freq": kw.get("freq", 130.0),
        "amp": kw.get("amp", 0.0),
        "window_ms": WINDOW_MS,
        "step_ms": STEP_MS,
        "zoom_w": zoom_w,
        "pb_fn": getattr(pb_fn, "__name__", str(pb_fn)),
    }, sort_keys=True)


def run_seeds(eng, seeds, pb_fn, out_dir="runs_switch", zoom_w=500, **kw):
    Path(out_dir).mkdir(exist_ok=True)
    switch_idx = int(kw.get("hold_ms", 5_000) / DT_MS)
    settings = _settings(pb_fn, zoom_w, kw)
    results, zooms, unverified = {}, {}, []

    for seed in seeds:
        seed = int(seed)
        path = Path(out_dir) / f"seed{seed}.npz"
        if path.exists():
            d = np.load(path)
            if "settings" in d.files:
                cached = str(d["settings"].item())
                if cached != settings:
                    raise ValueError(
                        f"{path} was written with different settings.\n"
                        f"  cached:  {cached}\n"
                        f"  current: {settings}\n"
                        f"Pass a different out_dir, or delete {out_dir}/ to re-simulate."
                    )
            else:
                unverified.append(seed)
            results[seed] = (d["times"], d["values"])
            if "zoom" in d.files:
                zooms[seed] = d["zoom"]
            continue

        vgi = run_switch(eng, seed=seed, **kw)
        times, values = sliding_pb(vgi, pb_fn)
        zoom = vgi[:, switch_idx - zoom_w:switch_idx + zoom_w]
        np.savez(path, times=times, values=values, zoom=zoom,
                 settings=np.array(settings))
        results[seed] = (times, values)
        zooms[seed] = zoom

    if unverified:
        print(f"warning: {len(unverified)} cached file(s) predate settings tracking "
              f"and were not checked: seeds {unverified}")
    return results, zooms


def summarize(results, switch_s=5.0):
    gaps, seps = [], []
    for t, v in results.values():
        b = v[t <= switch_s - HALF_WINDOW_S]
        a = v[t >= switch_s + HALF_WINDOW_S]
        gaps.append(100 * (a.mean() / b.mean() - 1))
        seps.append((a.mean() - b.mean()) / np.sqrt((b.var() + a.var()) / 2))
    return np.array(gaps), np.array(seps)


def detect_cp(times, values, thresh=7.0, min_size=4):
    for i in range(2 * min_size, len(values)):
        h = values[:i + 1]
        best = 0.0
        for p in range(min_size, len(h) - min_size + 1):
            a, b = h[:p], h[p:]
            s = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
            if s > 0:
                best = max(best, (b.mean() - a.mean()) / s)
        if best > thresh:
            return times[i] + HALF_WINDOW_S
    return np.nan


def detect_kmeans(times, values, min_gap=0.13, max_breaks=6, min_size=3):
    from sklearn.cluster import KMeans
    for i in range(2 * min_size, len(values)):
        hist = values[:i + 1].reshape(-1, 1)
        km = KMeans(2, n_init=10, random_state=0).fit(hist)
        lab, c = km.labels_, km.cluster_centers_.ravel()
        is_high = lab == int(np.argmax(c))
        if min(is_high.sum(), (~is_high).sum()) < min_size:
            continue
        if (c.max() - c.min()) / c.min() < min_gap:
            continue
        if (~is_high[np.argmax(is_high):]).sum() > max_breaks:
            continue
        return times[i] + HALF_WINDOW_S
    return np.nan


def evaluate(results, detector, switch_s=5.0, **kw):
    lat, false_alarm, missed = [], [], []
    for seed, (t, v) in sorted(results.items()):
        d = detector(t, v, **kw)
        if np.isnan(d):
            missed.append(seed)
        elif d < switch_s:
            false_alarm.append(seed)
        else:
            lat.append(d - switch_s)
    lat = np.array(lat)
    return {
        "ok": len(lat),
        "n": len(results),
        "false": false_alarm,
        "missed": missed,
        "latency": float(lat.mean()) if len(lat) else float("nan"),
        "sem": float(lat.std(ddof=1) / np.sqrt(len(lat))) if len(lat) > 1 else float("nan"),
    }


def report(label, r):
    print(f"{label:28s} ok {r['ok']:2d}/{r['n']}  false {r['false']}  "
          f"missed {r['missed']}  latency {r['latency']:.2f}s +/- {r['sem']:.2f}")


def plot_pb(times, values, switch_s=5.0, pd_start=0.3, pd_end=0.7,
            out_png="pd_switch_pb.png"):
    import matplotlib.pyplot as plt

    before = values[times <= switch_s - HALF_WINDOW_S]
    after = values[times >= switch_s + HALF_WINDOW_S]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axvspan(switch_s - HALF_WINDOW_S, switch_s + HALF_WINDOW_S, color="0.85", zorder=0)
    ax.plot(times, values, marker="o", ms=4, lw=1.8, color="tab:purple", zorder=3)
    ax.axvline(switch_s, ls="--", lw=1.5, color="tab:red", zorder=2)

    if len(before):
        ax.axhline(before.mean(), ls=":", lw=1.3, color="tab:blue")
        ax.text(times[0], before.mean(), f"  mean {before.mean():,.0f}",
                va="bottom", ha="left", color="tab:blue", fontsize=9)
    if len(after):
        ax.axhline(after.mean(), ls=":", lw=1.3, color="tab:orange")
        ax.text(times[-1], after.mean(), f"mean {after.mean():,.0f}  ",
                va="bottom", ha="right", color="tab:orange", fontsize=9)

    ax.annotate("window straddles\nthe switch",
                xy=(switch_s + HALF_WINDOW_S, ax.get_ylim()[1]),
                xytext=(switch_s + HALF_WINDOW_S + 0.6, ax.get_ylim()[1]),
                ha="left", va="top", fontsize=8, color="0.35")
    ax.text(switch_s, ax.get_ylim()[0], f"  pd {pd_start} -> {pd_end}",
            ha="left", va="bottom", color="tab:red", fontsize=10)

    if len(before) and len(after):
        pct = 100 * (after.mean() - before.mean()) / before.mean()
        ax.set_title(f"Beta-band power across a pd switch   "
                     f"(intensity {'+' if pct > 0 else ''}{pct:.0f}%)")
    else:
        ax.set_title("Beta-band power across a pd switch")

    ax.set_xlabel("time (s)")
    ax.set_ylabel(r"$P_\beta$  (13-35 Hz, 1000 ms window)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return fig


def plot_seeds(results, switch_s=5.0, pd_start=0.3, pd_end=0.7,
               out_png="pd_switch_seeds.png"):
    import matplotlib.pyplot as plt

    times = next(iter(results.values()))[0]
    stack = np.vstack([v for _, v in results.values()])
    mean, sd = stack.mean(0), stack.std(0)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axvspan(switch_s - HALF_WINDOW_S, switch_s + HALF_WINDOW_S, color="0.85", zorder=0)
    for t, v in results.values():
        ax.plot(t, v, lw=0.9, alpha=0.3, color="tab:purple", zorder=2)
    ax.fill_between(times, mean - sd, mean + sd, alpha=0.2, color="tab:purple", zorder=3)
    ax.plot(times, mean, lw=2.5, color="tab:purple", zorder=4)
    ax.axvline(switch_s, ls="--", lw=1.5, color="tab:red", zorder=5)
    ax.text(switch_s, ax.get_ylim()[0], f"  pd {pd_start} -> {pd_end}",
            ha="left", va="bottom", color="tab:red", fontsize=10)

    gaps, _ = summarize(results, switch_s)
    ax.set_title(f"Beta-band power across a pd switch, {len(results)} seeds   "
                 f"(intensity +{gaps.mean():.1f}% +/- {gaps.std(ddof=1)/np.sqrt(len(gaps)):.1f})")
    ax.set_xlabel("time (s)")
    ax.set_ylabel(r"$P_\beta$  (13-35 Hz, 1000 ms window)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return fig


def plot_zoom(zoom, neuron=0, out_png="pd_switch_zoom.png"):
    import matplotlib.pyplot as plt

    w = zoom.shape[1] // 2
    t = np.arange(-w, w) * DT_MS
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, zoom[neuron], lw=0.9, color="tab:purple")
    ax.axvline(0, ls="--", lw=1.4, color="tab:red")
    ax.set_xlabel("ms relative to the pd switch")
    ax.set_ylabel(f"GPi membrane voltage, neuron {neuron}")
    ax.set_title("Neural state is continuous across the switch")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return fig