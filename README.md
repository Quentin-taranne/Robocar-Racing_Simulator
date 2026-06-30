# Robocar Racing Simulator

Supervised learning pipeline for a racing simulator: collect driving data, clean it, train a small MLP per agent, test it back in the simulator.

> a small model with a big, clean and diverse dataset is better than a big model with a small dataset

## Pipeline

| # | Step | Command | Output |
|---|---|---|---|
| 1 | Launch the simulator | `open RacingSimulator.app` | — |
| 2 | Collect data | `scripts/run_client.sh` | `data/drive_*.csv` |
| 3 | Run EDA | `python3 scripts/eda_dataset.py` | `reports/data_eda.md` |
| 4 | *(optional)* Review sessions, mark crashes | `python3 scripts/review_session.py data/drive_0_*.csv` | `data_clean_exclusions.json` |
| 5 | Clean the dataset | `python3 scripts/clean_dataset.py --drop-stuck-segments --drop-reset-crashes` | `data_clean/*.csv` |
| 6 | Train a model per agent | `python robocar_client/train_model.py data_clean/drive_0_*.csv --agent-id 0 --input-source manette --out models/agent0.pt` | `models/*.pt`, `models/*.metrics.json` |
| 7 | Check the metrics | `cat models/agent0.metrics.json` | — |
| 8 | Test in the simulator | `python robocar_client/inference_client.py --pair 'Agent0?team=0:models/agent0.pt' ...` | — |

Exact commands for the current shipped models (`agent0`, `agent1`) are in [Step 6: Train](#step-6-train-a-model) below — the two agents don't use the same flags. Everything else in this README is detail and rationale for each step; read it when you need it, not to find the pipeline itself.

## Repository Layout

- `robocar_client/client.py` — manual driving client and data collection
- `robocar_client/data_logger.py` — streams driving frames to CSV, drops the last `--trim-seconds` before saving
- `robocar_client/train_model.py` — supervised training with saved evaluation metrics
- `robocar_client/ray_features.py` — shared raycast feature engineering (capped distance + "nothing detected" flag), used identically by training and inference
- `robocar_client/inference_client.py` — autonomous driving with a trained model
- `scripts/run_client.sh` — shortcut to launch manual driving
- `scripts/eda_dataset.py` — dataset EDA
- `scripts/plot_predictions.py` — visualize a trained model's throttle/steer predictions vs recorded actions
- `scripts/review_session.py` — frame-by-frame visual review (raycast reconstruction) to manually mark crashes/bad frames
- `scripts/clean_dataset.py` — dataset cleaning (manual exclusions, dedup, label-noise resolution, stuck/crash detection, sample weights)
- `scripts/train_agent.sh` — shortcut to train a model
- `scripts/run_inference.sh` — shortcut to run inference
- `data/` — collected raw CSV files
- `data_clean/` — cleaned CSV files produced by `scripts/clean_dataset.py`, used for training
- `data_clean_exclusions.json` — manually reviewed crash/bad-frame ranges, produced by `scripts/review_session.py`
- `models/` — trained models and exported metrics
- `reports/` — generated EDA and cleaning reports

## Prerequisites

- Conda or Miniforge for the reproducible `robocar39` environment
- `RacingSimulator.app` present at the repository root
- macOS accessibility permissions if you use global keyboard input

## Installation

### Reproduce the known working Conda environment

```bash
conda env create -f environment.yml
conda activate robocar39
python --version  # expected: 3.9.18
```

### Alternative lightweight venv

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r robocar_client/requirements.txt
```

ML-Agents `0.30.0` is not compatible with Python `>=3.11`.

---

## Step Details

### Step 1: Launch the Simulator

```bash
open RacingSimulator.app
```

### Step 2: Collect Data

```bash
python robocar_client/client.py \
  --env-path RacingSimulator.app \
  --agents-config robocar_client/agents.json \
  --base-port 5004
```

Or the shortcut: `scripts/run_client.sh` (supports the same behavior via environment variables, see below).

The last `--trim-seconds` (default `2.0`) of every session are held in memory and never written to disk when you stop recording — the moment right before you stop is almost always a crash or a bad decision, not clean driving. Override with `--trim-seconds 3` (or `TRIM_SECONDS=3 scripts/run_client.sh`) if needed.

<details>
<summary>Gamepad mapping, multi-agent collection, input debugging</summary>

**Gamepad default mapping**: left stick horizontal = steering, right trigger = throttle, left trigger = brake, A/Cross = throttle fallback, B/Circle = brake fallback.

```bash
INPUT_MODE=auto IDLE_THROTTLE=0.0 scripts/run_client.sh
```

If triggers/steering are mapped differently by macOS/SDL:

```bash
INPUT_MODE=gamepad GAMEPAD_STEER_AXIS=0 GAMEPAD_THROTTLE_AXIS=5 GAMEPAD_BRAKE_AXIS=4 scripts/run_client.sh
```

Use `GAMEPAD_INVERT_STEER=1` if steering is reversed. If the gamepad does not respond, inspect the raw SDL mapping: `python scripts/debug_gamepad.py`, then move the stick / press `L2`/`R2` and note which axis numbers change.

**Collect both agents at once**: `ALL_BEHAVIORS=1 scripts/run_client.sh` writes separate files per `agent_id` (`data/drive_0_*.csv`, `data/drive_1_*.csv`). Keep training one model per agent — they don't share an observation layout.

**Record straight into validation files**: `OUTPUT_DIR=validation_files scripts/run_client.sh` (default `data`) — useful for recording a deliberately clean run to use as validation (step 6) without sorting files afterward.

**Input not responding**: `INPUT_MODE=auto INPUT_DEBUG=1 GAMEPAD_DEBUG=1 IDLE_THROTTLE=0.0 scripts/run_client.sh` prints what the client reads and sends. Keep the small pygame window focused for `INPUT_MODE=auto`/`pygame`. For `INPUT_MODE=global`, grant Accessibility permission to the terminal app and restart it. If debug output stays at all-zero, the selected input mode isn't receiving OS events at all — that's a focus/permission problem, not a Robocar bug.

**Controlled collection example**:

```bash
python robocar_client/client.py \
  --env-path RacingSimulator.app --agents-config robocar_client/agents.json --base-port 5004 \
  --input-mode global --idle-throttle 0.0 --time-scale 1.5 --max-steps 5000
```

</details>

### Step 3: Run EDA

```bash
python3 scripts/eda_dataset.py
```

Outputs `reports/data_eda.md` and `reports/data_eda.json`. It answers: is the dataset big enough, structurally clean, and diverse enough to avoid learning only "go straight at full throttle"? Use these metrics, not just watching the car drive, to judge data quality — and prefer more varied data over a bigger model.

### Step 4 (optional): Visually Review Sessions and Mark Crashes

```bash
python3 scripts/review_session.py data/drive_0_*.csv
```

<details>
<summary>Why this step exists, and how to use it</summary>

If a session includes a crash or the car wedged against a wall, that segment has a perfectly *consistent* label (e.g. full throttle, zero steer, going nowhere) — nothing downstream flags it as noise, but the model will faithfully learn to repeat it. `reward` and `done` are constant in this dataset, so there is no telemetry signal to detect this automatically; this is the only step with actual access to what happened.

It reviews a session **frame by frame** (one CSV row = one frame) — the only way to catch something like "the car drove straight for half a second right before a turn it should have taken." For each file it opens a window with a throttle/steer overview (cursor = current frame, red = already marked) and a live top-down reconstruction of the current frame's raycasts (car at the center, each ray in its real direction with its measured distance, colored red=close to green=far).

Navigation, frame by frame:
- `→` / `←` — move one frame forward/backward (hold to flip through quickly)
- `d` (held) + `→`/`←` — every frame passed over while held gets marked for deletion (red); release to navigate without marking
- `u` — un-mark the current frame
- `c` — clear all marks for the current file
- `s` — save this file's marks and move to the next
- `n` — skip without saving
- `q` — quit (unsaved marks for the current file are lost)

Marked frames merge into `[start, end]` timestamp ranges, written to `data_clean_exclusions.json`. `scripts/clean_dataset.py` removes them **before** anything else. This step is optional — skip it if you trust the automatic heuristics below, or are iterating quickly and will review later.

</details>

### Step 5: Clean the Dataset

```bash
python3 scripts/clean_dataset.py --data-dir data --out-dir data_clean --report reports/data_clean.md --drop-stuck-segments --drop-reset-crashes
```

Train on `data_clean/`, never on `data/` directly. Outputs `data_clean/*.csv` (same schema, plus `input_source` and `sample_weight` columns) and `reports/data_clean.md` (per-file counts for every check below).

<details>
<summary>What gets cleaned, and why</summary>

Raw `data/` traces have three problems that hurt training directly: ~48% duplicate-observation rows (car sitting still), ~39% label noise (same observation, different recorded action — mostly keyboard jitter), and crashes recorded as if they were normal driving (see step 4). `scripts/clean_dataset.py` handles the first two automatically, plus two crash heuristics:

- **Source classification**: each CSV is tagged `clavier` (keyboard, discrete `{-1,0,1}` actions) or `manette` (gamepad, continuous actions).
- **Dedup**: drops consecutive frames where both observation and action are unchanged — no new signal.
- **Label-noise resolution**: when the same observation maps to different actions in a session, replaces them with their mean **only if the spread is small** (`<= 0.3`, treated as jitter). A genuine divergence (e.g. `-0.8` vs `+0.7` steer for the same observation) is left untouched — averaging it would invent a "go straight" label that was never driven.
- **Stuck/crash segments**: runs of >= 1s where the observation barely changes while throttle stays active — a car wedged against something. Listed always; `--drop-stuck-segments` removes them.
- **Reset/crash lead-ups (agent0 only)**: the simulator snaps the car to a fixed pose after a crash. The detector catches that jump and looks back up to 3s for a close raycast (`<= 30`) — if found, the frames between contact and reset are the bad decision that caused the crash. `--drop-reset-crashes` removes them. **Agent1 is excluded**: its narrower-FOV 36-ray sensor flickers a "nothing detected" sentinel during normal driving, producing jumps as large as real resets, with no reliable way found yet to tell them apart.
- **`sample_weight`**: rebalances steer/throttle bins per row, without dropping data. Computed but off by default (see step 6 — it hurt agent0 in practice).

On this project's data: stuck-segment detector flagged **11.4% of agent0's raw rows** vs **1.0% for agent1**; the reset-crash detector additionally caught **91 short lead-ups for agent0** that the stuck detector misses (crash-to-reset can happen in under a second). Both are heuristics on the observation alone — they miss crashes that don't freeze the raycasts or trigger a reset, which is what step 4 is for.

Known limitation: averaging small-spread label conflicts assumes the ambiguity really is noise, not a real (if subtle) behavioral difference — a limitation of regression-based behavior cloning in general, not a bug here.

</details>

### Step 6: Train a Model

Pin validation on dedicated "good run" sessions instead of the default last-20%-of-files split, so metrics stay comparable across retrains. Keep those sessions in their own folder (e.g. `validation_files/`), clean them the same way as `data/`, then pass **both** folders as input — `--val-files` only marks which already-loaded files are validation, it does not load them on its own:

```bash
python3 scripts/clean_dataset.py --data-dir validation_files --out-dir validation_files_clean \
  --report reports/validation_clean.md --drop-stuck-segments --drop-reset-crashes
```

```bash
# agent0
python robocar_client/train_model.py data_clean/drive_0_*.csv validation_files_clean/drive_0_*.csv \
  --agent-id 0 --input-source manette --split file --val-files validation_files_clean/drive_0_*.csv \
  --epochs 100 --batch-size 256 --hidden-size 128 \
  --out models/agent0.pt

# agent1
python robocar_client/train_model.py data_clean/drive_1_*.csv validation_files_clean/drive_1_*.csv \
  --agent-id 1 --input-source manette --expand-ray-features --add-ray-velocity --split file --val-files validation_files_clean/drive_1_*.csv \
  --epochs 200 --batch-size 256 --lr 0.0005 --hidden-size 256 \
  --out models/agent1.pt
```

Or via the shortcut script: `EPOCHS=100 HIDDEN=128 OUT=models/agent0.pt scripts/train_agent.sh data_clean/drive_0_*.csv --agent-id 0 --input-source manette`.

`models/*.metrics.json` stores the normalization stats — keep it next to the `.pt`, inference needs it. Hyperparameters above came from sweeping `--hidden-size`/`--lr`/`--batch-size`/`--weight-decay` and keeping the smallest, fastest-converging network that didn't lose accuracy. For agent1, `--lr 0.0005` (vs the `1e-3` default) clearly helped — turn-only steer MAE went from `~0.38` to `~0.28`; the same change made no difference for agent0, so it keeps the default. `--weight-decay` (Adam L2, default `0`), `--use-sample-weight`, and `--extreme-weight` (see below) were all tried for agent1 too; only `--add-ray-velocity` stuck.

**`--extreme-weight`** (available, not used in the shipped models) multiplies the training loss weight of rows with a hard brake or a sharp turn — `scripts/plot_predictions.py` showed the model never fully commits to these rare, decisive actions (MSE regression smooths a recorded full brake/full lock toward a "safe" middle value). It measurably sharpened commitment on exactly those frames, but cost enough normal-driving precision that it felt worse overall once tested live — reverted.

**`--add-ray-velocity`** appends, for each ray, the change in distance since the previous frame (computed in chronological order per session, before any split/shuffle). A plain MLP only sees one frame at a time — it cannot tell "this wall is closing in fast" from "this wall is closing in slowly" from a single distance reading, only from how that reading is changing. This was diagnosed concretely: recording agent1 live (`--log-dir`, see step 8) and replaying the crash through `scripts/review_session.py` showed the car holding `steer` near `0` for ~2.5s while an obstacle's raycast distance dropped steadily (149→117), then snapping to `-0.71` far too late — exactly the "takes the inside too hard" symptom. Checking the same kind of gradual-approach moments in the recorded human data showed steering ramping up much earlier (mean `|steer|` `0.09` in the first half of the approach, not near-zero), so the data already had the anticipation the model wasn't reproducing. Adding the per-ray closing speed as an explicit input dropped validation RMSE `0.203` → `0.184` and MAE `0.123` → `0.105` on the same data; only meaningful in closed-loop testing, since it targets a failure mode that doesn't show up the same way when replaying recorded (open-loop) trajectories.

**In-simulator status**: agent0 rarely crashes. Agent1 drives noticeably better on track 2 (after the `--lr` fix) but was still repeatedly clipping the inside of one specific left turn — the gradual-approach anticipation issue above — and still crashes on track 3. `--add-ray-velocity` targets the first; track 3 remains unexplained. CSVs don't currently record which track a session was driven on, so this can't yet be isolated as a training/validation split — collect and tag track-3-specific sessions (e.g. in the filename) before trying to fix that one with data rather than guessing.

<details>
<summary>Why <code>--input-source manette</code>, why <code>--expand-ray-features</code> only for agent1, explicit validation files, sample weights</summary>

**Why gamepad-only.** `data_clean/*.csv` contains both `clavier` and `manette` sessions, but the shipped models train on `manette` only (~130k rows for agent0, ~155k for agent1 — still plenty):
- gamepad actions are continuous, so the model learns an actual steering curve instead of being pulled toward three discrete keyboard taps.
- keyboard sessions have a much higher label-conflict rate (same observation, hard `-1`/`0`/`1` tap depending on reaction time) — exactly the noise that hurts a regression model most.
- on the same validation split, the gamepad-only model clearly beats the mean-regressor baseline on both throttle and steer RMSE; the mixed model didn't reliably.

Pass `--input-source clavier` if you have no gamepad data. Avoid `--input-source all` for production models — mixing reintroduces the label noise above.

**Why agent1 needs `--expand-ray-features`.** Both agents have much higher steer error on turns than on straight sections, but agent1 is consistently worse on turns than agent0 (steer MAE ~0.40 vs ~0.36) despite having essentially the same turn-row volume and balance (~16% of rows, same left/right split for both). Two structural reasons:
- `agents.json` gives agent0 a 180° FOV (10 rays) vs agent1's 48° (36 rays) — agent1 only "sees" a turn once almost on top of it, far less lead time than agent0.
- agent1's sensor returns a capped distance (~256, noisy tail to ~280) when a ray hits nothing; raw normalization treats that cap as just another continuous distance instead of a categorical "nothing detected" signal.

`--expand-ray-features` fixes the second point: it splits each raw ray distance into `[capped distance, "nothing detected" flag]` (`robocar_client/ray_features.py`), applied identically at training and inference time (the transform is recorded in `metrics.json` and replayed automatically by `inference_client.py`). On agent1 this took turn-only steer MAE from `0.401` to `0.383` at `--hidden-size 256` — real but modest, since the FOV limitation is the bigger factor and this doesn't fix that. It's agent1-only: agent0's sensor never clusters near a capped value, so the flag would be constant and useless there.

**Sample weights** (`--use-sample-weight`) rebalance steer/throttle bins via the `sample_weight` column. In practice this over-corrected for rare turns and made agent0 worse, so the shipped models don't use it — revisit if you collect a lot more recovery/turn data and the model still looks throttle-biased.

</details>

### Step 7: Read Evaluation Metrics

```bash
cat models/agent0.metrics.json
```

Check `val_metrics.mae`, `val_metrics.rmse`, `val_metrics.action_within_0.25`, and the `baselines` comparison. **If the model does not clearly beat the baselines, watching the car drive is not enough to call it good** — a model can improve RMSE while making worse driving decisions.

<details>
<summary>Baseline warnings, discrete vs continuous metrics, turn-balance check</summary>

With keyboard data, `exact_action_accuracy` is meaningful (discrete actions). With gamepad data it's `n/a` (continuous) — use `action_within_0.10`/`0.25`, MAE, RMSE instead.

Do not test a model in the simulator if training prints a baseline warning. Common failure: `val_metrics.rmse` beats `baselines.mean_regressor.rmse`, but `exact_action_accuracy` (or `action_within_0.25` for gamepad) is worse than the majority/mean baseline — the model predicts mostly one action (e.g. full throttle, no steering) and crashes.

Check whether the dataset has enough turns/recovery trajectories — if most rows are `(throttle=1.0, steer=0.0)`, the model can learn to drive straight and still look acceptable on RMSE:

```bash
python - <<'PY'
import csv, json, sys
from collections import Counter

counts = Counter()
for path in sys.argv[1:]:
    with open(path, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            counts[tuple(json.loads(row["action"])[:2])] += 1

print(counts.most_common())
PY data/drive_1_*.csv
```

</details>

### Step 8: Test a Trained Model

```bash
python robocar_client/inference_client.py \
  --pair 'Agent0?team=0:models/agent0.pt' \
  --env-path RacingSimulator.app \
  --agents-config robocar_client/agents.json \
  --base-port 5004
```

Both agents at once:

```bash
python robocar_client/inference_client.py \
  --pair 'Agent0?team=0:models/agent0.pt' \
  --pair 'Agent1?team=0:models/agent1.pt' \
  --env-path RacingSimulator.app \
  --agents-config robocar_client/agents.json \
  --base-port 5004
```

Or via the shortcut: `scripts/run_inference.sh --pair 'Agent0?team=0:models/agent0.pt'`. Add `--max-steps 3000` to limit the run.

**Record what the AI itself sees and decides** (e.g. to debug a specific crash spot): add `--log-dir debug_recordings`. Writes the same CSV format as `client.py` (obs + executed action), directly readable by `scripts/review_session.py` — unlike manual collection, nothing is trimmed off the end, so the crash itself is in the file. Drive/let it run up to and past the failure, then `Ctrl+C`.

**Visualize what a model actually predicts** (useful before blaming the simulator for "weird" driving):

```bash
python scripts/plot_predictions.py models/agent1.pt validation_files_clean/drive_1_*.csv \
  --agent-id 1 --input-source manette --expand-ray-features
```

Writes PNGs to `reports/predictions/`: predicted vs recorded throttle binned by turn sharpness (does the model actually brake for turns?), and a full predicted-vs-recorded timeline per file. On agent1 this showed the model never fully commits to a hard brake (rounds a recorded `0` down to `~0.3-0.4`). An output-smoothing fix was tried here and reverted — it introduced an aliasing bug (the smoothing state got mutated by the throttle clamp, causing throttle to ratchet up and stick at the ceiling) that tanked both agents' success rate; not worth re-attempting without a clearer payoff.

---

## Recommended Training Strategy

- Train one model per agent — don't mix agents with different sensor layouts (fov/nbRay).
- If collecting multiple agents at once, keep only sessions where each logged agent was actually driving valid lines; a shared command is a bad label for an agent that's already off-track or stuck.
- Always run `scripts/clean_dataset.py` before training — it cut this project's row count by ~33% before any agent-specific filtering.
- Recording includes crashes by default unless trimmed: `scripts/review_session.py` (step 4) is the most reliable way to remove them; `--drop-stuck-segments --drop-reset-crashes` (step 5) is a faster but imperfect fallback.
- Prefer `--input-source manette` for production models when you have gamepad data (see step 6 rationale).
- Keep the model small first — a `--hidden-size` sweep showed no benefit beyond 128-256 units here; data quality moved the metrics, not parameter count.
- Improve the dataset before scaling the model: more turns (both directions), more low-throttle situations, more recovery trajectories, less duplicated quasi-static driving.
- Use file-level validation (`--split file`, the default) to avoid inflated metrics from correlated consecutive rows.

## Common Files Produced

| File | Produced by |
|---|---|
| `data/drive_*.csv` | data collection (step 2) |
| `reports/data_eda.md`, `reports/data_eda.json` | EDA (step 3) |
| `data_clean_exclusions.json` | manual review (step 4) |
| `data_clean/drive_*.csv` | cleaning (step 5) |
| `reports/data_clean.md` | cleaning (step 5) |
| `models/*.pt`, `models/*.metrics.json` | training (step 6) |

## One Full Example

```bash
source .venv/bin/activate
open RacingSimulator.app
```

In another terminal:

```bash
source .venv/bin/activate
scripts/run_client.sh
```

After collecting data:

```bash
source .venv/bin/activate
python3 scripts/eda_dataset.py
python3 scripts/clean_dataset.py --data-dir data --out-dir data_clean --report reports/data_clean.md --drop-stuck-segments --drop-reset-crashes
python robocar_client/train_model.py \
  data_clean/drive_0_*.csv \
  --agent-id 0 --input-source manette \
  --epochs 100 --batch-size 256 --hidden-size 128 \
  --out models/agent0.pt
cat models/agent0.metrics.json
python robocar_client/inference_client.py \
  --pair 'Agent0?team=0:models/agent0.pt' \
  --env-path RacingSimulator.app \
  --agents-config robocar_client/agents.json \
  --base-port 5004
```
