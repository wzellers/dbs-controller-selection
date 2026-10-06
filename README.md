# DBS controller selection

Research code for a UNC CS project on deep brain stimulation for Parkinson's disease.

The question: a patient's symptom severity is not fixed — it changes with what they are
doing. If we train a separate stimulation controller for each severity level, does each
one genuinely specialize, and can we tell online which one to use?

This repo holds the per-severity model training done so far. Everything runs against a
Rubin–Terman style basal-ganglia–thalamic model in MATLAB, where severity is a parameter
`pd` from 0 (healthy) to 1 (severe).

---

## What's here

| File | What it is |
|---|---|
| `dbs_reward_model.ipynb` | **The main deliverable.** Neural network reward models trained at pd = 0.3, 0.6, 1.0, plus the cross-severity evaluation. |
| `dbs_rerun.ipynb` | Two separate pieces of work. Cells 0–29 are a TD3 re-run (abandoned, see below). Cells 30–54 are T3P bandit arm sweeps and runs at several severities. |
| `dbs_core.py` | Config, TD3 agent, replay buffer, MATLAB and mock environments, training loop. |
| `dbs_reward_model.py` | Data collection, reward-model fitting, cross-evaluation, action selection. |
| `dbs_t3p.py` | T3P bandit — arm grid, sweeps, pruning, the run loop. |
| `dbs_checks.py` | Offline and live environment checks run before any experiment. |
| `dbs_plots.py` | Training-curve and metric plots. |
| `helpers.py` | Biomarker and reward primitives (`calculate_pb`, `normalize_pb`). See note below. |
| `datasets/` | 400 sampled rows per severity, about 35 minutes of MATLAB each. |
| `matlab/` | The basal-ganglia–thalamic model: `bgn_init.m`, `bgn_step.m`, and `set_pd.m` for changing severity mid-simulation. |
| `gating/` | The model's 28 gating-variable functions, called by `bgn_step`. |

Modules are flat in the repo root because the notebooks import them directly
(`import dbs_core`). The MATLAB model sits in `matlab/` and `gating/`, both of which are
added to the MATLAB path at startup.

---

## Approach: reward models instead of policy learning

A deep-RL policy (TD3) was tried first and abandoned. It failed for a specific, diagnosable
reason: its untrained actor only ever proposed amplitudes between roughly 1600 and 3500
µA, so it never sampled the low-amplitude region where the good settings turned out to be.
It was training on half the action space and lost to fixed open-loop stimulation.

The reward-model approach removes that failure mode by construction:

- **Actions are sampled uniformly by us**, so training data covers the whole space.
- **The correct answer is measured, not bootstrapped.** Apply a setting, record the
  biomarker and the energy used, compute the reward. That is the label.
- **MATLAB runs once per severity, not once per architecture.** The same 400-sample dataset
  trains any number of networks in seconds, which turns "which network size is best" from a
  multi-hour experiment into a supervised-learning question.

The controller is then `argmax` over candidate actions under the learned model.

---

## Results

### Each model is best on the severity it was trained for

Cross-evaluation at hidden size 32 — every model scored against every severity's held-out
data. Values are R², the fraction of reward variance explained.

| model \ data | pd = 0.3 | pd = 0.6 | pd = 1.0 |
|---|---|---|---|
| **trained at 0.3** | **0.497** | 0.243 | 0.051 |
| **trained at 0.6** | 0.250 | **0.523** | 0.366 |
| **trained at 1.0** | −0.057 | 0.294 | **0.570** |

The diagonal wins everywhere and performance falls off with distance. Neighbouring
severities transfer partially; distant ones do not transfer at all. Negative R² means worse
than predicting the mean.

### Network size does not matter

Validation R² across hidden sizes, averaged over 3 seeds:

| hidden units | parameters | pd = 0.3 | pd = 0.6 | pd = 1.0 |
|---|---|---|---|---|
| 16 | 401 | 0.445 ± 0.121 | 0.474 ± 0.029 | 0.494 ± 0.066 |
| 32 | 1,313 | 0.457 ± 0.118 | 0.496 ± 0.043 | 0.497 ± 0.068 |
| 64 | 4,673 | 0.414 ± 0.113 | 0.469 ± 0.020 | 0.506 ± 0.064 |
| 128 | 17,537 | 0.443 ± 0.120 | 0.493 ± 0.021 | 0.514 ± 0.064 |

Flat within error bars across a 44× range in parameter count. Whatever limits accuracy here
is not model capacity.

### The important caveat: specialization does not reach the action

Despite the R² matrix, **the models almost all select the same action** — 180 Hz at the
lowest amplitude — regardless of which severity they were trained on. At hidden size 32 the
0.3 and 1.0 models pick it in 100% of states and the 0.6 model in 68%.

So the specialization is real in *reward prediction* but does not become different *control
decisions*. Two things drive this, and both need addressing before the result means what it
appears to mean:

1. **180 Hz was the top of the frequency grid.** This project has produced an apparent
   optimum sitting on a grid edge several times (1000 µA, then 130 Hz, then 80 Hz, then
   180 Hz). An optimum at the boundary usually means the true one is outside the box.
2. **The energy penalty dominates.** In the collected data, mean reward falls monotonically
   with amplitude at every severity, so the reward function as weighted prefers minimal
   stimulation. The models are not wrong about the objective; the objective may not be
   the one we want.

Read the R² table as evidence that a selector *has something to select on*, not as evidence
that the controllers behave differently yet.

### T3P bandit (in `dbs_rerun.ipynb`, cells 30+)

The bandit from the group's earlier published work, applied per severity. At pd = 1.0 over
a 31-arm grid it converged to 155 Hz / 1000 µA. A fixed 155 Hz / 450 µA setting scored
−0.007 on a mild brain against −0.064 on a severe one, a gap of about +0.057, consistent
across three seeds.

---

## Running it

```bash
# MATLAB engine is only installed under this interpreter
/opt/anaconda3/bin/python -c "import matlab.engine; print('ok')"
```

Start Jupyter from the repo root so the notebooks can import the modules and MATLAB can
find `bgn_vars.mat`. The code adds `matlab/` and `gating/` to the MATLAB path itself. Simulations must run **serially** — `bgn_vars.mat` is a single shared
state file rewritten every step, and concurrent MATLAB engines corrupt each other.

`dbs_reward_model.ipynb` runs top to bottom. Section 3 collects data and takes roughly 35
minutes per severity; section 3b reloads from `datasets/` so you only pay that once.

---

## Known issues

- **Integer overflow in the energy term (fixed).** Runs before module version 2.6.0
  computed `dbs ** 2` in the array's native integer dtype, which wrapped negative and
  silently corrupted the RMS. All array arithmetic is now float64. Any result predating
  2.6.0 should be discarded.
- **Grid-edge optima.** Check whether a selected action sits at the boundary of the search
  grid in either dimension before believing it.
- **`dbs_rerun.ipynb` is a working notebook,** not a clean artifact. The TD3 half documents
  an approach that was abandoned; it is kept for the record.

---

## Not included

- **`bgn_vars.mat`** — the model's working state. It is regenerated by `bgn_init` on every
  run, and the copy on disk reaches several hundred megabytes, so it is deliberately kept
  out of version control.
- **`runs_*/` caches** — regenerable experiment output.
