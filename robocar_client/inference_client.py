"""Inference client: charge un modèle entraîné et contrôle la voiture automatiquement."""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, Set, Tuple

import numpy as np
import torch
from mlagents_envs.environment import ActionTuple, UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)

from data_logger import DataLogger
from ray_features import compute_ray_velocity_step, expand_ray_hit_features


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
        "--map-signatures",
        type=Path,
        default=None,
        help=(
            "JSON de signatures de départ par map (produit par scripts/extract_map_signature.py). "
            "Si fourni, détecte et affiche automatiquement la map au début de chaque épisode."
        ),
    )
    p.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help=(
            "Enregistre les observations et actions executees pendant l'inference (meme format que "
            "client.py, relisible par review_session.py/plot_predictions.py). Desactive par defaut."
        ),
    )
    p.add_argument(
        "--hidden",
        type=int,
        default=None,
        help="Taille cachee du MLP. Si omise, on essaie de la lire depuis model.metrics.json.",
    )
    return p.parse_args()


def resolve_hidden_size(model_path: Path, cli_hidden: int | None) -> int:
    if cli_hidden is not None:
        return cli_hidden

    metrics_path = model_path.with_suffix(".metrics.json")
    if metrics_path.exists():
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            hidden = int(payload["model"]["hidden_size"])
            return hidden
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass

    return HIDDEN_SIZE


def load_model_metadata(model_path: Path) -> dict[str, object]:
    metrics_path = model_path.with_suffix(".metrics.json")
    if not metrics_path.exists():
        return {}
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def resolve_obs_normalization(metadata: dict[str, object], input_dim: int) -> tuple[np.ndarray, np.ndarray] | None:
    model_meta = metadata.get("model")
    if not isinstance(model_meta, dict):
        return None
    normalization = model_meta.get("obs_normalization")
    if not isinstance(normalization, dict) or not normalization.get("enabled"):
        return None

    mean = np.asarray(normalization.get("mean", []), dtype=np.float32)
    std = np.asarray(normalization.get("std", []), dtype=np.float32)
    if mean.shape != (input_dim,) or std.shape != (input_dim,):
        raise ValueError(
            f"Normalisation invalide dans les metriques: mean={mean.shape}, std={std.shape}, input_dim={input_dim}"
        )
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def resolve_ray_hit_feature(metadata: dict[str, object]) -> dict[str, object] | None:
    model_meta = metadata.get("model")
    if not isinstance(model_meta, dict):
        return None
    ray_hit_feature = model_meta.get("ray_hit_feature")
    if not isinstance(ray_hit_feature, dict) or not ray_hit_feature.get("enabled"):
        return None
    return {
        "n_rays": int(ray_hit_feature["n_rays"]),
        "no_hit_distance": float(ray_hit_feature["no_hit_distance"]),
    }


def resolve_ray_velocity_feature(metadata: dict[str, object]) -> dict[str, object] | None:
    model_meta = metadata.get("model")
    if not isinstance(model_meta, dict):
        return None
    ray_velocity_feature = model_meta.get("ray_velocity_feature")
    if not isinstance(ray_velocity_feature, dict) or not ray_velocity_feature.get("enabled"):
        return None
    return {"n_rays": int(ray_velocity_feature["n_rays"])}


def load_map_signatures(path: Path) -> Dict[str, np.ndarray]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {name: np.asarray(obs, dtype=np.float32) for name, obs in raw.items()}


def detect_map(obs: np.ndarray, signatures: Dict[str, np.ndarray]) -> str:
    best_name = min(signatures, key=lambda n: float(np.linalg.norm(obs - signatures[n])))
    best_dist = float(np.linalg.norm(obs - signatures[best_name]))
    return f"{best_name} (L2={best_dist:.1f})"


def flatten_obs_batch(decision_steps) -> list[np.ndarray]:
    return [
        np.concatenate([obs.flatten() for obs in decision_steps[agent_id].obs]).astype(np.float32)
        for agent_id in decision_steps.agent_id
    ]


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
    controllers: Dict[
        str,
        Tuple[MLP, dict[str, object] | None, dict[str, object] | None, tuple[np.ndarray, np.ndarray] | None],
    ] = {}
    for beh, spec in env.behavior_specs.items():
        if beh not in behavior_to_model:
            print(f"[WARN] behavior {beh} ignoré (pas de modèle associé)")
            continue
        flat_dim = int(sum(np.prod(obs_spec.shape) for obs_spec in spec.observation_specs))
        model_path = behavior_to_model[beh]
        metadata = load_model_metadata(model_path)
        hidden_size = resolve_hidden_size(model_path, args.hidden)
        ray_hit_feature = resolve_ray_hit_feature(metadata)
        ray_velocity_feature = resolve_ray_velocity_feature(metadata)
        effective_dim = flat_dim
        if ray_hit_feature:
            effective_dim += ray_hit_feature["n_rays"]
        if ray_velocity_feature:
            effective_dim += ray_velocity_feature["n_rays"]
        expected_input_dim = metadata.get("model", {}).get("input_dim") if isinstance(metadata.get("model"), dict) else None
        if expected_input_dim is not None and int(expected_input_dim) != effective_dim:
            raise ValueError(
                f"{beh}: dimension observation simulateur (apres feature engineering eventuel)={effective_dim}, "
                f"modele={expected_input_dim}. Verifiez agents.json et le modele utilise."
            )
        model = MLP(input_dim=effective_dim, output_dim=spec.action_spec.continuous_size, hidden=hidden_size)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        obs_normalization = resolve_obs_normalization(metadata, effective_dim)
        controllers[beh] = (model, ray_hit_feature, ray_velocity_feature, obs_normalization)
        print(
            f"[INFO] {beh} -> {model_path} | obs dim={flat_dim} (modele: {effective_dim}) | hidden={hidden_size} | "
            f"ray_hit_feature={ray_hit_feature is not None} | ray_velocity_feature={ray_velocity_feature is not None} | "
            f"normalize_obs={obs_normalization is not None} | actions cont.: {spec.action_spec.continuous_size}"
        )

    if not controllers:
        raise SystemExit("Aucun model associé à un behavior existant.")

    map_signatures: Dict[str, np.ndarray] = {}
    if args.map_signatures is not None:
        map_signatures = load_map_signatures(args.map_signatures)
        print(f"[INFO] {len(map_signatures)} signature(s) de map chargées: {list(map_signatures.keys())}")

    step_count = 0
    episode = 0
    _first_step = True
    with ExitStack() as stack:
        loggers: Dict[int, DataLogger] = {}

        def logger_for(agent_id: int) -> DataLogger:
            if agent_id not in loggers:
                # trim_seconds=0 : a l'inverse de la collecte manuelle, on veut justement
                # voir les derniers instants avant un crash, pas les jeter.
                loggers[agent_id] = stack.enter_context(DataLogger(args.log_dir, agent_id=agent_id, trim_seconds=0.0))
            return loggers[agent_id]

        previous_raw_obs: Dict[str, np.ndarray] = {}
        # Agents dont le prochain step est le premier frame d'un nouvel épisode
        map_detect_pending: Dict[str, Set[int]] = {}

        try:
            while True:
                for beh, spec in env.behavior_specs.items():
                    if beh not in controllers:
                        continue
                    model, ray_hit_feature, ray_velocity_feature, obs_normalization = controllers[beh]
                    decision_steps, terminal_steps = env.get_steps(beh)
                    if len(decision_steps) == 0:
                        continue
                    raw_obs_batch = np.stack(flatten_obs_batch(decision_steps))

                    if map_signatures:
                        pending = map_detect_pending.get(beh, set())
                        for row, agent_id in enumerate(decision_steps.agent_id):
                            if _first_step or int(agent_id) in pending:
                                detected = detect_map(raw_obs_batch[row], map_signatures)
                                print(f"[MAP] episode {episode} | {beh} agent {agent_id} -> {detected}")
                                pending.discard(int(agent_id))
                    _first_step = False

                    velocity = None
                    if ray_velocity_feature is not None:
                        velocity = compute_ray_velocity_step(
                            raw_obs_batch, previous_raw_obs.get(beh), ray_velocity_feature["n_rays"]
                        )
                    # Toujours stocker une copie explicite, jamais une reference : raw_obs_batch
                    # n'est mute par rien plus bas, mais on l'isole volontairement de toute mutation
                    # future pour ne pas reproduire le bug d'aliasing du lissage retire precedemment.
                    previous_raw_obs[beh] = raw_obs_batch.copy()

                    obs_batch = raw_obs_batch
                    if ray_hit_feature is not None:
                        obs_batch = expand_ray_hit_features(
                            obs_batch, ray_hit_feature["n_rays"], ray_hit_feature["no_hit_distance"]
                        )
                    if velocity is not None:
                        obs_batch = np.concatenate([obs_batch, velocity], axis=-1)
                    if obs_normalization is not None:
                        mean, std = obs_normalization
                        obs_batch = (obs_batch - mean) / std
                    obs_tensor = torch.tensor(obs_batch, dtype=torch.float32)
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

                    if args.log_dir is not None:
                        for row, agent_id in enumerate(decision_steps.agent_id):
                            logger_for(int(agent_id)).log(
                                episode=episode,
                                step=step_count,
                                agent_id=int(agent_id),
                                reward=float(decision_steps[agent_id].reward),
                                done=False,
                                action=acts[row],
                                obs=raw_obs_batch[row],
                            )

                    action_tuple = ActionTuple(continuous=acts)
                    env.set_actions(beh, action_tuple)

                    if map_signatures and len(terminal_steps) > 0:
                        pending = map_detect_pending.setdefault(beh, set())
                        for agent_id in terminal_steps.agent_id:
                            pending.add(int(agent_id))

                    if args.log_dir is not None and len(terminal_steps) > 0:
                        terminal_obs = np.stack(
                            [
                                np.concatenate([o.flatten() for o in terminal_steps[agent_id].obs])
                                for agent_id in terminal_steps.agent_id
                            ]
                        ).astype(np.float32)
                        for row, agent_id in enumerate(terminal_steps.agent_id):
                            logger_for(int(agent_id)).log(
                                episode=episode,
                                step=step_count,
                                agent_id=int(agent_id),
                                reward=float(terminal_steps[agent_id].reward),
                                done=True,
                                action=acts[0] if len(acts) else np.zeros(2, dtype=np.float32),
                                obs=terminal_obs[row],
                            )
                        episode += 1

                env.step()
                step_count += 1
                if args.max_steps and step_count >= args.max_steps:
                    print("Max steps atteint, arrêt.")
                    break
        except KeyboardInterrupt:
            print("Arrêt utilisateur")
        finally:
            env.close()
            for logger in loggers.values():
                print(f"Trace enregistree dans {logger.filepath}")


if __name__ == "__main__":
    main()
