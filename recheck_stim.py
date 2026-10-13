"""Re-score the cached under-stimulation runs with pair-averaged P_beta.

No simulation. Reads runs_switch/ and every runs_stim_*/ directory already on disk.

Why: P_beta is computed on a 1000 ms window stepped 250 ms. At high stimulation
amplitude the series picks up a period-2 component, which inflates the within-segment
standard deviation without moving either segment mean. Separation is a ratio of the two,
so it collapses even though the switch is still there. Averaging adjacent windows
cancels a period-2 component exactly.
"""

import numpy as np
from pathlib import Path

import pd_switch as pdsw


def load_cached(out_dir):
    out = {}
    for p in sorted(Path(out_dir).glob("seed*.npz"), key=lambda q: int(q.stem[4:])):
        d = np.load(p)
        out[int(p.stem[4:])] = (d["times"], d["values"])
    return out


def pair_average(results):
    out = {}
    for s, (t, v) in results.items():
        m = (len(v) // 2) * 2
        out[s] = ((t[:m:2] + t[1:m:2]) / 2.0, (v[:m:2] + v[1:m:2]) / 2.0)
    return out


def calibrate(cal, grid=(3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0), verbose=True):
    paired, clean = pair_average(cal), []
    for th in grid:
        r = pdsw.evaluate(paired, pdsw.detect_cp, thresh=th, min_size=3)
        if not r["false"] and not r["missed"]:
            clean.append(th)
        if verbose:
            print(f"  thresh {th:4.1f}  ok {r['ok']:2d}/{r['n']}  "
                  f"false {r['false']}  missed {r['missed']}")
    if not clean:
        return None
    return float(np.median(clean))


def run(raw_thresh=7.0):
    cal = {s: v for s, v in load_cached("runs_switch").items() if s <= 10}
    if not cal:
        print("no runs_switch/ cache found — run the unstimulated seeds first")
        return

    print(f"calibrating paired threshold on unstimulated seeds 1-{max(cal)}")
    thresh = calibrate(cal)
    if thresh is None:
        print("no threshold gave a clean calibration; widen the grid")
        return
    print(f"\nfrozen paired threshold: {thresh}   "
          f"(midpoint of the clean range, chosen without seeing any stimulated run)\n")

    dirs = sorted(p for p in Path(".").glob("runs_stim_*") if p.is_dir())
    if not dirs:
        print("no runs_stim_*/ caches found")
        return

    print(f"{'run':<34}{'n':>4}{'sep raw':>9}{'sep pair':>10}{'gain':>7}"
          f"{'det raw':>10}{'det pair':>10}")
    print("-" * 84)
    for d in dirs:
        raw = load_cached(d)
        if not raw:
            continue
        par = pair_average(raw)
        sr = pdsw.summarize(raw)[1]
        sp = pdsw.summarize(par)[1]
        rr = pdsw.evaluate(raw, pdsw.detect_cp, thresh=raw_thresh)
        rp = pdsw.evaluate(par, pdsw.detect_cp, thresh=thresh, min_size=3)
        print(f"{d.name:<34}{len(raw):>4}{sr.mean():9.2f}{sp.mean():10.2f}"
              f"{sp.mean() / sr.mean():6.2f}x"
              f"{rr['ok']:7d}/{rr['n']:<3}{rp['ok']:7d}/{rp['n']:<3}")

    print("\nsep = separation in within-segment sd, mean over brains")
    print(f"det raw uses the tuned raw threshold {raw_thresh}; "
          f"det pair uses {thresh}, frozen above")
    return thresh
