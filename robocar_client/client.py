"""Client minimal pour piloter le simulateur et collecter des données."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from mlagents_envs.environment import ActionTuple, UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)

from data_logger import DataLogger
from input_manager import GlobalKeyboardController, PygameKeyboardController


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Client Robocar minimal")
    parser.add_argument(
        "--env-path",
        type=Path,
        default=Path("RacingSimulator.app"),
        help="Chemin vers l'app Unity (.app). Laissez vide si vous lancez déjà le binaire.",
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=5004,
        help="Port utilisé par le simulateur",
    )
    parser.add_argument(
        "--behavior-name",
        type=str,
        default=None,
        help="Nom du behavior à contrôler (sinon le premier trouvé).",
    )
    parser.add_argument(
        "--all-behaviors",
        action="store_true",
        help="Contrôler tous les behaviors disponibles avec la même commande clavier.",
    )
    parser.add_argument(
        "--input-mode",
        choices=["pygame", "global"],
        default="pygame",
        help="pygame (fenêtre focus) ou global (pynput, nécessite autorisation accessibilité macOS).",
    )
    parser.add_argument(
        "--agents-config",
        type=Path,
        default=Path("agents.json"),
        help="Chemin vers le fichier JSON de configuration des agents (fov / nbRay)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Dossier où enregistrer les traces",
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="Facteur de vitesse du simulateur (1.0 = temps réel)",
    )
    parser.add_argument(
        "--idle-throttle",
        type=float,
        default=0.3,
        help="Accélération par défaut (0-1) quand aucune touche n'est pressée pour permettre au véhicule d'avancer.",
    )
    parser.add_argument(
        "--action-order",
        choices=["steer-throttle", "throttle-steer"],
        default="throttle-steer",
        help="Ordre des actions continues (2 dims): steer-throttle ou throttle-steer.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Nombre maximal de steps (0 = infini)",
    )
    parser.add_argument(
        "--no-graphics",
        action="store_true",
        help="Lancer Unity sans rendu graphique (utile pour l'entraînement)",
    )
    return parser.parse_args()


def build_action_vector(
    action_spec,
    steer: float,
    throttle: float,
    brake: float,
    idle: float,
    order: str,
) -> np.ndarray:
    """Adapte la commande clavier au format attendu par ML-Agents (continuous)."""
    throttle = np.clip(max(throttle, idle), 0.0, 1.0)
    steer = np.clip(steer, -1.0, 1.0)
    brake = np.clip(brake, 0.0, 1.0)

    continuous = np.zeros(action_spec.continuous_size, dtype=np.float32)
    if action_spec.continuous_size == 1:
        continuous[0] = throttle  # on pousse au moins l'accélération
    elif action_spec.continuous_size >= 2:
        if order == "throttle-steer":
            continuous[0] = throttle
            continuous[1] = steer
        else:  # steer-throttle
            continuous[0] = steer
            continuous[1] = throttle
        if action_spec.continuous_size > 2:
            continuous[2] = brake
    return continuous


def build_discrete_actions(action_spec, steer: float, throttle: float, brake: float, n_agents: int) -> np.ndarray:
    """
    Heuristique de mapping clavier -> actions discrètes.
    On essaye d'utiliser au mieux le nombre d'options disponibles.
    """
    branches = action_spec.discrete_branches
    disc = np.zeros((n_agents, action_spec.discrete_size), dtype=np.int32)

    if action_spec.discrete_size >= 2 and branches[0] >= 3:
        # Deux branches : direction puis accélération/frein
        steer_cmd = 1  # neutre
        if steer < -0.2:
            steer_cmd = 0  # gauche
        elif steer > 0.2:
            steer_cmd = min(2, branches[0] - 1)  # droite si dispo
        disc[:, 0] = steer_cmd

        accel_cmd = 0  # neutre
        if throttle > 0.1:
            accel_cmd = 1  # accélère
        elif brake > 0.1 and branches[1] >= 3:
            accel_cmd = 2  # freine
        disc[:, 1] = accel_cmd

    elif action_spec.discrete_size == 1:
        size = branches[0]
        cmd = 0
        if size >= 5:
            if throttle > 0.1:
                cmd = 1
            elif brake > 0.1:
                cmd = 2
            elif steer < -0.2:
                cmd = 3
            elif steer > 0.2:
                cmd = 4
        elif size == 4:
            # 0 neutre,1 accel,2 gauche,3 droite
            if throttle > 0.1:
                cmd = 1
            elif steer < -0.2:
                cmd = 2
            elif steer > 0.2:
                cmd = 3
        elif size == 3:
            # 0 gauche,1 neutre,2 droite
            if steer < -0.2:
                cmd = 0
            elif steer > 0.2:
                cmd = 2
            else:
                cmd = 1
        elif size == 2:
            # 0 neutre,1 accélère
            if throttle > 0.1:
                cmd = 1
        disc[:, 0] = cmd

    # sinon, on laisse à zéro (neutre)
    return disc


def main() -> None:
    args = parse_args()

    engine_channel = EngineConfigurationChannel()
    engine_channel.set_configuration_parameters(time_scale=args.time_scale)

    additional_args: list[str] = []
    if args.agents_config and args.agents_config.exists():
        additional_args += ["--config-path", str(args.agents_config.resolve())]

    # Unity accepte soit un .app (macOS) soit le binaire direct. On laisse UnityEnvironment
    # résoudre automatiquement le bon exécutable à l'intérieur du .app.
    env = UnityEnvironment(
        file_name=str(args.env_path) if args.env_path else None,
        base_port=args.base_port,
        no_graphics=args.no_graphics,
        side_channels=[engine_channel],
        additional_args=additional_args,
        timeout_wait=120,
    )

    controller = (
        PygameKeyboardController()
        if args.input_mode == "pygame"
        else GlobalKeyboardController()
    )
    env.reset()

    available = list(env.behavior_specs.keys())
    print(f"Behaviors disponibles: {available}")
    if args.behavior_name and args.all_behaviors:
        raise ValueError("--behavior-name et --all-behaviors sont exclusifs.")
    behavior_names = available if args.all_behaviors else [args.behavior_name or available[0]]
    for behavior_name in behavior_names:
        spec = env.behavior_specs[behavior_name]
        print(
            f"Connecté à {behavior_name} | obs: {len(spec.observation_specs)} | "
            f"actions cont.: {spec.action_spec.continuous_size} | actions disc.: {spec.action_spec.discrete_size}"
        )
        if spec.action_spec.discrete_size:
            print(f"Actions discrètes {behavior_name}: branches={spec.action_spec.discrete_branches}")

    step_count = 0
    episode = 0

    with ExitStack() as stack:
        loggers: dict[int, DataLogger] = {}

        def logger_for(agent_id: int) -> DataLogger:
            if agent_id not in loggers:
                loggers[agent_id] = stack.enter_context(DataLogger(args.output_dir, agent_id=agent_id))
            return loggers[agent_id]

        try:
            while True:
                behavior_steps = {
                    behavior_name: env.get_steps(behavior_name)
                    for behavior_name in behavior_names
                }
                active_count = sum(len(decision_steps) for decision_steps, _ in behavior_steps.values())
                terminal_count = sum(len(terminal_steps) for _, terminal_steps in behavior_steps.values())
                if step_count % 50 == 0:
                    print(f"step {step_count}: agents actifs={active_count} terminés={terminal_count}")

                # Une commande clavier pour tous les agents à ce step
                steer, throttle, brake = controller.poll()

                action_vectors: dict[str, np.ndarray] = {}
                for behavior_name, (decision_steps, _) in behavior_steps.items():
                    spec = env.behavior_specs[behavior_name]
                    action_vec = build_action_vector(
                        spec.action_spec,
                        steer,
                        throttle,
                        brake,
                        idle=args.idle_throttle,
                        order=args.action_order,
                    )
                    action_vectors[behavior_name] = action_vec

                    n_agents = len(decision_steps)
                    if n_agents > 0:
                        actions = np.tile(action_vec, (n_agents, 1))
                    else:
                        actions = np.zeros((0, spec.action_spec.continuous_size), dtype=np.float32)

                    discrete = None
                    if spec.action_spec.discrete_size > 0:
                        discrete = build_discrete_actions(
                            spec.action_spec, steer=steer, throttle=throttle, brake=brake, n_agents=n_agents
                        )

                    env.set_actions(
                        behavior_name,
                        ActionTuple(continuous=actions if actions.size else None, discrete=discrete),
                    )

                # Log décisions
                for behavior_name, (decision_steps, _) in behavior_steps.items():
                    action_vec = action_vectors[behavior_name]
                    for agent_id in decision_steps.agent_id:
                        obs_list = decision_steps[agent_id].obs
                        flat_obs = np.concatenate([o.flatten() for o in obs_list])
                        reward = float(decision_steps[agent_id].reward)
                        agent_id_int = int(agent_id)
                        logger_for(agent_id_int).log(
                            episode=episode,
                            step=step_count,
                            agent_id=agent_id_int,
                            reward=reward,
                            done=False,
                            action=action_vec,
                            obs=flat_obs,
                        )

                # Étape suivante
                env.step()
                step_count += 1

                # Log fins d'épisode
                if terminal_count:
                    for behavior_name, (_, terminal_steps) in behavior_steps.items():
                        action_vec = action_vectors[behavior_name]
                        for agent_id in terminal_steps.agent_id:
                            obs_list = terminal_steps[agent_id].obs
                            flat_obs = np.concatenate([o.flatten() for o in obs_list])
                            reward = float(terminal_steps[agent_id].reward)
                            agent_id_int = int(agent_id)
                            logger_for(agent_id_int).log(
                                episode=episode,
                                step=step_count,
                                agent_id=agent_id_int,
                                reward=reward,
                                done=True,
                                action=action_vec,
                                obs=flat_obs,
                            )
                    episode += 1

                if args.max_steps and step_count >= args.max_steps:
                    print("Limite de steps atteinte, arrêt propre…")
                    break
                print(f"Episode {episode} | Step {step_count} | Agents actifs: {active_count}", end="\r")

        except KeyboardInterrupt:
            print("Arrêt utilisateur")
        finally:
            env.close()
            for logger in loggers.values():
                print(f"Données enregistrées dans {logger.filepath}")


if __name__ == "__main__":
    main()
