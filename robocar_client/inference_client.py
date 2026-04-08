"""Inference client: charge un modèle entraîné et contrôle la voiture automatiquement."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from mlagents_envs.environment import ActionTuple, UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)


HIDDEN_SIZE = 256
ACTION_ORDER = "throttle-steer"
DEFAULT_ENV_PATH = "RacingSimulator.app"
DEFAULT_AGENTS_CONFIG = "robocar_client/agents.json"
DEFAULT_BASE_PORT = 5004


class MLP(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int = HIDDEN_SIZE) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, output_dim),
            torch.nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference client Robocar (multi-behavior)")
    p.add_argument(
        "--pair",
        action="append",
        required=True,
        help="Mapping behavior:model.pt, ex: --pair 'Agent0?team=0:models/a.pt'",
    )
    p.add_argument("--env-path", type=Path, default=Path(DEFAULT_ENV_PATH))
    p.add_argument("--agents-config", type=Path, default=Path(DEFAULT_AGENTS_CONFIG))
    p.add_argument("--base-port", type=int, default=DEFAULT_BASE_PORT)
    p.add_argument("--time-scale", type=float, default=1.0)
    p.add_argument("--no-graphics", action="store_true")
    p.add_argument(
        "--action-order",
        choices=["throttle-steer", "steer-throttle"],
        default=ACTION_ORDER,
        help="Ordre attendu par l'environnement pour (throttle, steer)",
    )
    p.add_argument("--min-throttle", type=float, default=0.1, help="Throttle plancher appliqué sur la prédiction")
    p.add_argument(
        "--max-throttle",
        type=float,
        default=0.5,
        help="Throttle maximum après prédiction (utile si l'agent reste plein gaz)",
    )
    p.add_argument(
        "--steer-sign",
        type=float,
        default=1.0,
        help="Mettre -1 si la direction est inversée pour un agent",
    )
    p.add_argument("--max-steps", type=int, default=0, help="0 = infini")
    p.add_argument(
        "--hidden", type=int, default=HIDDEN_SIZE, help="Taille cachée du MLP (doit matcher le modèle entraîné)"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Parse behavior:model pairs
    behavior_to_model: Dict[str, Path] = {}
    for pair in args.pair:
        if ":" not in pair:
            raise SystemExit(f"Format attendu pour --pair: behavior:model.pt (reçu: {pair})")
        beh, mod = pair.split(":", 1)
        behavior_to_model[beh] = Path(mod)

    engine_channel = EngineConfigurationChannel()
    engine_channel.set_configuration_parameters(time_scale=args.time_scale)

    additional_args: list[str] = []
    if args.agents_config and args.agents_config.exists():
        additional_args += ["--config-path", str(args.agents_config.resolve())]

    env = UnityEnvironment(
        file_name=str(args.env_path) if args.env_path else None,
        base_port=args.base_port,
        no_graphics=args.no_graphics,
        side_channels=[engine_channel],
        additional_args=additional_args,
        timeout_wait=120,
    )
    env.reset()

    # Load models per behavior
    controllers: Dict[str, Tuple[MLP, int]] = {}
    for beh, spec in env.behavior_specs.items():
        if beh not in behavior_to_model:
            print(f"[WARN] behavior {beh} ignoré (pas de modèle associé)")
            continue
        obs_shape = spec.observation_specs[0].shape
        flat_dim = int(np.prod(obs_shape))
        model = MLP(input_dim=flat_dim, output_dim=spec.action_spec.continuous_size, hidden=args.hidden)
        model.load_state_dict(torch.load(behavior_to_model[beh], map_location="cpu"))
        model.eval()
        controllers[beh] = (model, flat_dim)
        print(
            f"[INFO] {beh} -> {behavior_to_model[beh]} | obs dim={flat_dim} | actions cont.: {spec.action_spec.continuous_size}"
        )

    if not controllers:
        raise SystemExit("Aucun model associé à un behavior existant.")

    step_count = 0
    try:
        while True:
            for beh, spec in env.behavior_specs.items():
                if beh not in controllers:
                    continue
                model, _ = controllers[beh]
                decision_steps, terminal_steps = env.get_steps(beh)
                if len(decision_steps) == 0:
                    continue
                obs_batch = [decision_steps[aid].obs[0].flatten() for aid in decision_steps.agent_id]
                obs_tensor = torch.tensor(np.stack(obs_batch), dtype=torch.float32)
                with torch.no_grad():
                    acts = model(obs_tensor).cpu().numpy()

                acts = np.clip(acts, -1.0, 1.0)
                if acts.shape[1] >= 2:
                    throttle_idx, steer_idx = (0, 1) if args.action_order == "throttle-steer" else (1, 0)
                    throttle = np.clip(acts[:, throttle_idx] + args.min_throttle, 0.0, args.max_throttle)
                    steer = np.clip(acts[:, steer_idx] * args.steer_sign, -1.0, 1.0)
                    acts[:, throttle_idx] = throttle
                    acts[:, steer_idx] = steer
                elif acts.shape[1] == 1:
                    acts[:, 0] = np.clip(np.abs(acts[:, 0]) + args.min_throttle, 0.0, args.max_throttle)

                action_tuple = ActionTuple(continuous=acts)
                # if acts.shape[1] >= 2:
                    # if args.action_order == "throttle-steer":
                    #     throttle_idx, steer_idx = 0, 1
                    # else:
                    #     steer_idx, throttle_idx = 0, 1
                    # steer_pred = np.clip(acts[:, steer_idx], -1.0, 1.0)
                    # acts[:, throttle_idx] = 0.25    # throttle fixe
                    # acts[:, steer_idx] = steer_pred # on ne garde que la direction du modèle


                env.set_actions(beh, action_tuple)

            env.step()
            step_count += 1
            if args.max_steps and step_count >= args.max_steps:
                print("Max steps atteint, arrêt.")
                break
    except KeyboardInterrupt:
        print("Arrêt utilisateur")
    finally:
        env.close()


if __name__ == "__main__":
    main()
