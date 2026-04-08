#!/usr/bin/env bash
# Entraîne un modèle supervisé. Ajuste les variables pour éviter la répétition.
# Exemple :
#   ACTION_ORDER=steer-throttle scripts/train_agent.sh data/drive_0_*.csv --agent-id 0 --out models/agent0.pt

EPOCHS=${EPOCHS:-300}
BATCH=${BATCH:-256}
LR=${LR:-1e-3}
SEED=${SEED:-42}
OUT=${OUT:-models/steering_mlp.pt}
ACTION_ORDER=${ACTION_ORDER:-throttle-steer}  # ordre des actions dans les CSV

set -euo pipefail

python robocar_client/train_model.py \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH" \
  --lr "$LR" \
  --seed "$SEED" \
  --out "$OUT" \
  "$@"
