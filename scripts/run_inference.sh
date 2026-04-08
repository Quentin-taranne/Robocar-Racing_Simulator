#!/usr/bin/env bash
# Inférence multi-behavior. Renseigne les paires behavior:model ci-dessous pour éviter de les répéter.
# Exemple : PAIRS_DEFAULT=("Agent0?team=0:models/agent0.pt" "Agent1?team=0:models/agent1.pt")

set -eo pipefail

PAIRS_DEFAULT=()  # modifie ici si tu veux des paires par défaut
PAIRS=("${PAIRS_DEFAULT[@]}")  # array toujours définie, même vide

ENV_PATH=${ENV_PATH:-RacingSimulator.app}
AGENTS_CONFIG=${AGENTS_CONFIG:-$(pwd)/robocar_client/agents.json}
BASE_PORT=${BASE_PORT:-5004}
MIN_THROTTLE=${MIN_THROTTLE:-0.1}
MAX_STEPS=${MAX_STEPS:-0}

cmd=(python robocar_client/inference_client.py
  --env-path "$ENV_PATH"
  --agents-config "$AGENTS_CONFIG"
  --base-port "$BASE_PORT"
  --min-throttle "$MIN_THROTTLE"
)

[[ "$MAX_STEPS" -gt 0 ]] && cmd+=(--max-steps "$MAX_STEPS")

for p in "${PAIRS[@]}"; do
  cmd+=(--pair "$p")
done

# Ajoute les arguments supplémentaires passés à ce script
cmd+=("$@")

"${cmd[@]}"
