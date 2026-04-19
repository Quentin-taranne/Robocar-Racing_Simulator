# Robocar Racing Simulator

Supervised learning pipeline for a racing simulator:
- collect driving trajectories from manual control
- validate the dataset with EDA
- train a small MLP on CSV traces
- test the model back in the simulator

The current repo is built around one simple principle:

> a small model with a big, clean and diverse dataset is better than a big model with a small dataset

## Repository Layout

- `robocar_client/client.py`: manual driving client and data collection
- `robocar_client/train_model.py`: supervised training with saved evaluation metrics
- `robocar_client/inference_client.py`: autonomous driving with a trained model
- `scripts/run_client.sh`: shortcut to launch manual driving
- `scripts/train_agent.sh`: shortcut to train a model
- `scripts/run_inference.sh`: shortcut to run inference
- `scripts/eda_dataset.py`: dataset EDA
- `data/`: collected CSV files
- `models/`: trained models and exported metrics
- `reports/`: generated EDA reports

## Prerequisites

- Python `3.10`
- `RacingSimulator.app` present at the repository root
- macOS accessibility permissions if you use global keyboard input

## Installation

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r robocar_client/requirements.txt
```

## Quick Workflow

1. Launch the simulator.
2. Drive manually and collect data into `data/`.
3. Run EDA and check dataset quality.
4. Train a model per agent.
5. Read metrics from `models/*.metrics.json`.
6. Test the trained model in the simulator.

## 1. Launch the Simulator

```bash
open RacingSimulator.app
```

## 2. Collect Data

### Direct command

```bash
python robocar_client/client.py \
  --env-path RacingSimulator.app \
  --agents-config robocar_client/agents.json \
  --base-port 5004
```

### Recommended shortcut

```bash
scripts/run_client.sh
```

### Controlled collection example

```bash
python robocar_client/client.py \
  --env-path RacingSimulator.app \
  --agents-config robocar_client/agents.json \
  --base-port 5004 \
  --input-mode global \
  --idle-throttle 0.0 \
  --time-scale 1.5 \
  --max-steps 5000
```

### Useful environment overrides for `scripts/run_client.sh`

```bash
INPUT_MODE=global IDLE_THROTTLE=0.0 scripts/run_client.sh
```

```bash
BEHAVIOR_NAME='Agent0?team=0' scripts/run_client.sh
```

Collected trajectories are saved as CSV files in `data/`.

## 3. Run EDA

```bash
python3 scripts/eda_dataset.py
```

Generated outputs:
- `reports/data_eda.md`
- `reports/data_eda.json`

The EDA is meant to answer:
- is the dataset big enough for a first prototype?
- is it structurally clean?
- is it diverse enough to avoid learning only "go straight at full throttle"?

Current quality expectations:
- use metrics, not only visual inspection of the car
- keep the dataset big, clean and diverse
- prefer more varied data before increasing model size

## 4. Train a Model

Training exports:
- the model weights, for example `models/agent0.pt`
- evaluation metrics, for example `models/agent0.metrics.json`

By default, training uses validation by CSV file instead of a pure random split. This is intentional: random row-level validation is too optimistic when consecutive samples are highly correlated.

### Train agent 0

```bash
python robocar_client/train_model.py \
  data/drive_0_*.csv \
  --agent-id 0 \
  --epochs 20 \
  --batch-size 256 \
  --hidden-size 128 \
  --out models/agent0.pt
```

### Train agent 1

```bash
python robocar_client/train_model.py \
  data/drive_1_*.csv \
  --agent-id 1 \
  --epochs 20 \
  --batch-size 256 \
  --hidden-size 128 \
  --out models/agent1.pt
```

### Train with the shortcut script

```bash
HIDDEN=128 OUT=models/agent0.pt scripts/train_agent.sh \
  data/drive_0_*.csv \
  --agent-id 0
```

### Explicit validation files

```bash
python robocar_client/train_model.py \
  data/drive_0_*.csv \
  --agent-id 0 \
  --split file \
  --val-files data/drive_0_20260408-155743.csv \
  --out models/agent0.pt
```

## 5. Read Evaluation Metrics

```bash
cat models/agent0.metrics.json
```

Important metrics to inspect:
- `val_metrics.mae`
- `val_metrics.rmse`
- `val_metrics.exact_action_accuracy`
- `val_metrics.throttle_accuracy`
- `val_metrics.steer_accuracy`
- baseline comparison under `baselines`

If the model does not clearly beat simple baselines, watching the car drive is not enough to call it good.

## 6. Test a Trained Model

### Direct inference command

```bash
python robocar_client/inference_client.py \
  --pair 'Agent0?team=0:models/agent0.pt' \
  --env-path RacingSimulator.app \
  --agents-config robocar_client/agents.json \
  --base-port 5004
```

### Test both agents

```bash
python robocar_client/inference_client.py \
  --pair 'Agent0?team=0:models/agent0.pt' \
  --pair 'Agent1?team=0:models/agent1.pt' \
  --env-path RacingSimulator.app \
  --agents-config robocar_client/agents.json \
  --base-port 5004
```

### Limit the run

```bash
python robocar_client/inference_client.py \
  --pair 'Agent0?team=0:models/agent0.pt' \
  --env-path RacingSimulator.app \
  --agents-config robocar_client/agents.json \
  --base-port 5004 \
  --max-steps 3000
```

### Use the shortcut script

```bash
scripts/run_inference.sh \
  --pair 'Agent0?team=0:models/agent0.pt'
```

```bash
scripts/run_inference.sh \
  --pair 'Agent0?team=0:models/agent0.pt' \
  --pair 'Agent1?team=0:models/agent1.pt'
```

## Recommended Training Strategy

- Train one model per agent instead of mixing agents with different sensor layouts.
- Keep the model small first, for example `--hidden-size 128`.
- Improve the dataset before scaling the model:
  - more left and right turns
  - more low-throttle situations
  - more recovery trajectories
  - less duplicated quasi-static sequences
- Use file-level validation to avoid inflated metrics.

## Common Files Produced

- data collection: `data/drive_*.csv`
- EDA report: `reports/data_eda.md`
- EDA summary: `reports/data_eda.json`
- trained model: `models/*.pt`
- training metrics: `models/*.metrics.json`

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
python robocar_client/train_model.py \
  data/drive_0_*.csv \
  --agent-id 0 \
  --epochs 20 \
  --batch-size 256 \
  --hidden-size 128 \
  --out models/agent0.pt
cat models/agent0.metrics.json
python robocar_client/inference_client.py \
  --pair 'Agent0?team=0:models/agent0.pt' \
  --env-path RacingSimulator.app \
  --agents-config robocar_client/agents.json \
  --base-port 5004
```
