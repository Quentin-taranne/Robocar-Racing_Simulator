#!/usr/bin/env bash
# Lance le client manuel. Pré-requis : conda env actif (robocar39) et RacingSimulator.app présent.
# Modifie simplement les variables ci-dessous pour éviter de répéter les options.

ENV_PATH=${ENV_PATH:-RacingSimulator.app} # doesn't work without this parameter
AGENTS_CONFIG=${AGENTS_CONFIG:-$(pwd)/robocar_client/agents.json}
OUTPUT_DIR=${OUTPUT_DIR:-data}     # data ou validation_files pour enregistrer directement comme run de validation
BASE_PORT=${BASE_PORT:-5004}
ACTION_ORDER=${ACTION_ORDER:-throttle-steer}
INPUT_MODE=${INPUT_MODE:-auto}     # auto, global, pygame ou gamepad
INPUT_DEBUG=${INPUT_DEBUG:-0}
BEHAVIOR_NAME=${BEHAVIOR_NAME:-}   # ex: "Agent0?team=0" ou "Agent1?team=0"
ALL_BEHAVIORS=${ALL_BEHAVIORS:-0}  # 1 = même input pour tous les behaviors
IDLE_THROTTLE=${IDLE_THROTTLE:-0.0}
GAMEPAD_INDEX=${GAMEPAD_INDEX:-0}
GAMEPAD_STEER_AXIS=${GAMEPAD_STEER_AXIS:-0}
GAMEPAD_THROTTLE_AXIS=${GAMEPAD_THROTTLE_AXIS:-5}
GAMEPAD_BRAKE_AXIS=${GAMEPAD_BRAKE_AXIS:-4}
GAMEPAD_THROTTLE_BUTTON=${GAMEPAD_THROTTLE_BUTTON:-0}
GAMEPAD_BRAKE_BUTTON=${GAMEPAD_BRAKE_BUTTON:-1}
GAMEPAD_DEADZONE=${GAMEPAD_DEADZONE:-0.12}
GAMEPAD_INVERT_STEER=${GAMEPAD_INVERT_STEER:-0}
GAMEPAD_DEBUG=${GAMEPAD_DEBUG:-0}

set -euo pipefail

extra_args=()
[[ "$ALL_BEHAVIORS" == "1" ]] && extra_args+=(--all-behaviors)
[[ "$INPUT_DEBUG" == "1" ]] && extra_args+=(--input-debug)
[[ "$GAMEPAD_INVERT_STEER" == "1" ]] && extra_args+=(--gamepad-invert-steer)
[[ "$GAMEPAD_DEBUG" == "1" ]] && extra_args+=(--gamepad-debug)

python robocar_client/client.py \
  --env-path "$ENV_PATH" \
  --agents-config "$AGENTS_CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --base-port "$BASE_PORT" \
  --action-order "$ACTION_ORDER" \
  --input-mode "$INPUT_MODE" \
  --idle-throttle "$IDLE_THROTTLE" \
  ${extra_args[@]+"${extra_args[@]}"} \
  ${BEHAVIOR_NAME:+--behavior-name "$BEHAVIOR_NAME"} \
  "$@"
