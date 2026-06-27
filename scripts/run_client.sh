#!/usr/bin/env bash
# Lance le client manuel. Pré-requis : conda env actif (robocar39) et RacingSimulator.app présent.
# Modifie simplement les variables ci-dessous pour éviter de répéter les options.

ENV_PATH=${ENV_PATH:-RacingSimulator.app} # doesn't work without this parameter
AGENTS_CONFIG=${AGENTS_CONFIG:-$(pwd)/robocar_client/agents.json}
BASE_PORT=${BASE_PORT:-5004}
ACTION_ORDER=${ACTION_ORDER:-throttle-steer}
INPUT_MODE=${INPUT_MODE:-global}   # global = pas besoin du focus pygame
BEHAVIOR_NAME=${BEHAVIOR_NAME:-}   # ex: "Agent0?team=0" ou "Agent1?team=0"
ALL_BEHAVIORS=${ALL_BEHAVIORS:-0}  # 1 = même clavier pour tous les behaviors
IDLE_THROTTLE=${IDLE_THROTTLE:-0.0}

set -euo pipefail

extra_args=()
[[ "$ALL_BEHAVIORS" == "1" ]] && extra_args+=(--all-behaviors)

python robocar_client/client.py \
  --env-path "$ENV_PATH" \
  --agents-config "$AGENTS_CONFIG" \
  --base-port "$BASE_PORT" \
  --action-order "$ACTION_ORDER" \
  --input-mode "$INPUT_MODE" \
  --idle-throttle "$IDLE_THROTTLE" \
  "${extra_args[@]}" \
  ${BEHAVIOR_NAME:+--behavior-name "$BEHAVIOR_NAME"} \
  "$@"
