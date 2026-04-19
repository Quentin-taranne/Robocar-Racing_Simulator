#!/usr/bin/env bash
# Entraîne un modèle supervisé. Ajuste les variables pour éviter la répétition.
# Exemple :
#   ACTION_ORDER=steer-throttle scripts/train_agent.sh data/drive_0_*.csv --agent-id 0 --out models/agent0.pt

EPOCHS=${EPOCHS:-300}
BATCH=${BATCH:-256}
LR=${LR:-1e-3}
HIDDEN=${HIDDEN:-128}
SEED=${SEED:-42}
OUT=${OUT:-models/steering_mlp.pt}
METRICS_OUT=${METRICS_OUT:-}
ACTION_ORDER=${ACTION_ORDER:-throttle-steer}  # ordre des actions dans les CSV

set -euo pipefail

cmd=(python robocar_client/train_model.py
  --epochs "$EPOCHS" \
  --batch-size "$BATCH" \
  --hidden-size "$HIDDEN" \
  --lr "$LR" \
  --seed "$SEED" \
  --out "$OUT" \
)

[[ -n "$METRICS_OUT" ]] && cmd+=(--metrics-out "$METRICS_OUT")
cmd+=("$@")

"${cmd[@]}"
