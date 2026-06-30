"""Entrainement supervise avec vraies metriques d'evaluation."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from ray_features import (
    NO_HIT_DISTANCE,
    compute_ray_velocity_batch,
    expand_ray_hit_features,
    load_agent_n_rays,
)

HIDDEN_SIZE = 256
ACTION_ORDER = "throttle-steer"  # ordre impose : col0 throttle, col1 steer
DEFAULT_AGENTS_CONFIG = Path("robocar_client/agents.json")


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int = HIDDEN_SIZE) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
            nn.Tanh(),  # actions bornees entre -1 et 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class SessionData:
    path: Path
    obs: np.ndarray
    actions: np.ndarray
    weights: np.ndarray


@dataclass
class DatasetSplit:
    train_obs: np.ndarray
    train_actions: np.ndarray
    train_weights: np.ndarray
    val_obs: np.ndarray
    val_actions: np.ndarray
    val_weights: np.ndarray
    train_files: list[str]
    val_files: list[str]
    split_name: str


@dataclass
class ObsNormalization:
    mean: np.ndarray
    std: np.ndarray


def build_obs_normalization(train_obs: np.ndarray) -> ObsNormalization:
    mean = train_obs.mean(axis=0).astype(np.float32)
    std = train_obs.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return ObsNormalization(mean=mean, std=std)


def apply_obs_normalization(obs: np.ndarray, normalization: ObsNormalization | None) -> np.ndarray:
    if normalization is None:
        return obs.astype(np.float32)
    return ((obs - normalization.mean) / normalization.std).astype(np.float32)


def reorder_actions(actions: np.ndarray) -> np.ndarray:
    if actions.shape[1] >= 2 and ACTION_ORDER == "throttle-steer":
        c0_min, c0_max = actions[:, 0].min(), actions[:, 0].max()
        c1_min, c1_max = actions[:, 1].min(), actions[:, 1].max()
        if c0_min < -0.5 and c0_max <= 1.0 and c1_min >= -0.1 and c1_max > 0.5:
            actions = actions[:, [1, 0]]
    return actions


def load_sessions(
    paths: Sequence[Path],
    agent_id: int | None = None,
    input_source: str = "all",
    expand_ray_features: bool = False,
    add_ray_velocity: bool = False,
    agents_config: Path = DEFAULT_AGENTS_CONFIG,
) -> list[SessionData]:
    sessions: list[SessionData] = []
    for path in paths:
        df = pd.read_csv(path)
        if agent_id is not None:
            df = df[df["agent_id"] == agent_id]
        if input_source != "all" and "input_source" in df.columns:
            df = df[df["input_source"] == input_source]
        if df.empty:
            continue
        # Les lignes restent dans l'ordre chronologique d'origine (filtres ci-dessus = masques
        # booleens, jamais de tri) : necessaire pour que la vitesse de rapprochement ci-dessous
        # soit calculee entre pas reellement consecutifs, pas entre lignes sans rapport.
        obs = np.stack(df["obs"].apply(json.loads).values).astype(np.float32)
        actions = np.stack(df["action"].apply(json.loads).values).astype(np.float32)
        actions = reorder_actions(actions)
        n_rays = None
        if expand_ray_features or add_ray_velocity:
            file_agent_id = int(df["agent_id"].iloc[0])
            n_rays = load_agent_n_rays(agents_config, file_agent_id)
            if n_rays is None:
                raise ValueError(
                    f"{path}: --expand-ray-features/--add-ray-velocity demande mais nbRay introuvable "
                    f"pour agent_id={file_agent_id} dans {agents_config}."
                )
        velocity = compute_ray_velocity_batch(obs, n_rays) if add_ray_velocity else None
        if expand_ray_features:
            obs = expand_ray_hit_features(obs, n_rays)
        if velocity is not None:
            obs = np.concatenate([obs, velocity], axis=-1)
        if "sample_weight" in df.columns:
            weights = df["sample_weight"].to_numpy(dtype=np.float32)
        else:
            weights = np.ones(len(df), dtype=np.float32)
        sessions.append(SessionData(path=path, obs=obs, actions=actions, weights=weights))
    if not sessions:
        raise ValueError("Aucun echantillon apres filtrage (paths/agent_id/input_source).")
    return sessions


def stack_sessions(sessions: Sequence[SessionData]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs = np.concatenate([session.obs for session in sessions], axis=0)
    actions = np.concatenate([session.actions for session in sessions], axis=0)
    weights = np.concatenate([session.weights for session in sessions], axis=0)
    return obs.astype(np.float32), actions.astype(np.float32), weights.astype(np.float32)


def split_by_file(
    sessions: Sequence[SessionData],
    *,
    val_ratio: float,
    val_files: Sequence[Path] | None = None,
) -> DatasetSplit:
    if val_files:
        val_names = {Path(path).name for path in val_files}
        train_sessions = [session for session in sessions if session.path.name not in val_names]
        val_sessions = [session for session in sessions if session.path.name in val_names]
    else:
        ordered = sorted(sessions, key=lambda session: session.path.name)
        if len(ordered) < 2:
            raise ValueError("Le split par fichier requiert au moins 2 CSV. Utiliser --split random sinon.")
        val_count = max(1, math.ceil(len(ordered) * val_ratio))
        if val_count >= len(ordered):
            val_count = len(ordered) - 1
        train_sessions = ordered[:-val_count]
        val_sessions = ordered[-val_count:]

    if not train_sessions or not val_sessions:
        raise ValueError("Split invalide: train/validation vide. Verifier --val-files ou --val-ratio.")

    train_obs, train_actions, train_weights = stack_sessions(train_sessions)
    val_obs, val_actions, val_weights = stack_sessions(val_sessions)
    return DatasetSplit(
        train_obs=train_obs,
        train_actions=train_actions,
        train_weights=train_weights,
        val_obs=val_obs,
        val_actions=val_actions,
        val_weights=val_weights,
        train_files=[session.path.name for session in train_sessions],
        val_files=[session.path.name for session in val_sessions],
        split_name="file",
    )


def split_random(
    sessions: Sequence[SessionData],
    *,
    val_ratio: float,
    seed: int,
) -> DatasetSplit:
    obs, actions, weights = stack_sessions(sessions)
    dataset = TensorDataset(
        torch.tensor(obs, dtype=torch.float32),
        torch.tensor(actions, dtype=torch.float32),
        torch.tensor(weights, dtype=torch.float32),
    )
    train_len = int(len(dataset) * (1.0 - val_ratio))
    train_len = min(max(1, train_len), len(dataset) - 1)
    val_len = len(dataset) - train_len
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=generator)

    train_obs = train_ds.dataset.tensors[0][train_ds.indices].cpu().numpy()
    train_actions = train_ds.dataset.tensors[1][train_ds.indices].cpu().numpy()
    train_weights = train_ds.dataset.tensors[2][train_ds.indices].cpu().numpy()
    val_obs = val_ds.dataset.tensors[0][val_ds.indices].cpu().numpy()
    val_actions = val_ds.dataset.tensors[1][val_ds.indices].cpu().numpy()
    val_weights = val_ds.dataset.tensors[2][val_ds.indices].cpu().numpy()

    file_names = [session.path.name for session in sessions]
    return DatasetSplit(
        train_obs=train_obs,
        train_actions=train_actions,
        train_weights=train_weights,
        val_obs=val_obs,
        val_actions=val_actions,
        val_weights=val_weights,
        train_files=file_names,
        val_files=file_names,
        split_name="random",
    )


def build_split(
    sessions: Sequence[SessionData],
    *,
    split: str,
    val_ratio: float,
    val_files: Sequence[Path],
    seed: int,
) -> DatasetSplit:
    if split == "file":
        return split_by_file(sessions, val_ratio=val_ratio, val_files=val_files)
    if val_files:
        raise ValueError("--val-files n'est supporte qu'avec --split file.")
    return split_random(sessions, val_ratio=val_ratio, seed=seed)


def build_loader(
    obs: np.ndarray,
    actions: np.ndarray,
    weights: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(obs, dtype=torch.float32),
        torch.tensor(actions, dtype=torch.float32),
        torch.tensor(weights, dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    per_sample = ((pred - target) ** 2).mean(dim=1)
    return (per_sample * weights).sum() / weights.sum()


def snap_predictions(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    snapped = pred.copy()
    if target.shape[1] >= 1:
        unique_throttle = set(np.unique(target[:, 0]).tolist())
        if unique_throttle.issubset({0.0, 1.0}):
            snapped[:, 0] = (snapped[:, 0] >= 0.5).astype(np.float32)
    if target.shape[1] >= 2:
        unique_steer = set(np.unique(target[:, 1]).tolist())
        if unique_steer.issubset({-1.0, 0.0, 1.0}):
            steer = np.zeros(len(snapped), dtype=np.float32)
            steer[snapped[:, 1] <= -0.5] = -1.0
            steer[snapped[:, 1] >= 0.5] = 1.0
            snapped[:, 1] = steer
    return snapped


def is_discrete_dim(values: np.ndarray) -> bool:
    unique_values = set(np.unique(values).round(6).tolist())
    return unique_values.issubset({-1.0, 0.0, 1.0})


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, object]:
    diff = pred - target
    abs_diff = np.abs(diff)
    sq_diff = diff ** 2

    mae_per_dim = abs_diff.mean(axis=0)
    mse_per_dim = sq_diff.mean(axis=0)
    rmse_per_dim = np.sqrt(mse_per_dim)

    target_mean = target.mean(axis=0)
    ss_res = ((target - pred) ** 2).sum(axis=0)
    ss_tot = ((target - target_mean) ** 2).sum(axis=0)
    r2_per_dim: list[float | None] = []
    for res, total in zip(ss_res, ss_tot):
        if total <= 1e-12:
            r2_per_dim.append(None)
        else:
            r2_per_dim.append(float(1.0 - (res / total)))

    metrics: dict[str, object] = {
        "mse": float(sq_diff.mean()),
        "rmse": float(np.sqrt(sq_diff.mean())),
        "mae": float(abs_diff.mean()),
        "mae_per_dim": [float(value) for value in mae_per_dim],
        "rmse_per_dim": [float(value) for value in rmse_per_dim],
        "r2_per_dim": r2_per_dim,
    }

    snapped = snap_predictions(pred, target)
    if target.shape[1] >= 1:
        metrics["throttle_within_0.10"] = float(np.mean(abs_diff[:, 0] <= 0.10))
        metrics["throttle_within_0.25"] = float(np.mean(abs_diff[:, 0] <= 0.25))
        if is_discrete_dim(target[:, 0]):
            metrics["throttle_accuracy"] = float(np.mean(snapped[:, 0] == target[:, 0]))
        else:
            metrics["throttle_accuracy"] = None
    if target.shape[1] >= 2:
        metrics["steer_within_0.10"] = float(np.mean(abs_diff[:, 1] <= 0.10))
        metrics["steer_within_0.25"] = float(np.mean(abs_diff[:, 1] <= 0.25))
        if is_discrete_dim(target[:, 1]):
            metrics["steer_accuracy"] = float(np.mean(snapped[:, 1] == target[:, 1]))
        else:
            metrics["steer_accuracy"] = None
    if target.shape[1] >= 2 and is_discrete_dim(target[:, 0]) and is_discrete_dim(target[:, 1]):
        metrics["exact_action_accuracy"] = float(np.mean(np.all(snapped[:, :2] == target[:, :2], axis=1)))
        metrics["action_within_0.10"] = float(np.mean(np.all(abs_diff[:, :2] <= 0.10, axis=1)))
        metrics["action_within_0.25"] = float(np.mean(np.all(abs_diff[:, :2] <= 0.25, axis=1)))
    elif target.shape[1] >= 2:
        metrics["exact_action_accuracy"] = None
        metrics["action_within_0.10"] = float(np.mean(np.all(abs_diff[:, :2] <= 0.10, axis=1)))
        metrics["action_within_0.25"] = float(np.mean(np.all(abs_diff[:, :2] <= 0.25, axis=1)))
    else:
        metrics["exact_action_accuracy"] = (
            float(np.mean(np.all(snapped == target, axis=1))) if is_discrete_dim(target[:, 0]) else None
        )
        metrics["action_within_0.10"] = float(np.mean(np.all(abs_diff <= 0.10, axis=1)))
        metrics["action_within_0.25"] = float(np.mean(np.all(abs_diff <= 0.25, axis=1)))

    return metrics


def collect_predictions(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, float]:
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    total_weighted_loss = 0.0
    total_weight = 0.0

    model.eval()
    with torch.no_grad():
        for xb, yb, wb in loader:
            pred = model(xb)
            batch_weight = wb.sum().item()
            total_weighted_loss += weighted_mse(pred, yb, wb).item() * batch_weight
            total_weight += batch_weight
            preds.append(pred.cpu().numpy())
            targets.append(yb.cpu().numpy())

    pred_np = np.concatenate(preds, axis=0)
    target_np = np.concatenate(targets, axis=0)
    return pred_np, target_np, total_weighted_loss / max(1e-12, total_weight)


def evaluate_model(model: nn.Module, loader: DataLoader) -> dict[str, object]:
    pred, target, loss = collect_predictions(model, loader)
    metrics = compute_metrics(pred, target)
    metrics["loss_mse"] = float(loss)
    return metrics


def build_baseline_metrics(train_actions: np.ndarray, val_actions: np.ndarray) -> dict[str, dict[str, object]]:
    mean_pred = np.repeat(train_actions.mean(axis=0, keepdims=True), len(val_actions), axis=0)
    majority_action = np.array(
        max(
            ((tuple(row.tolist()), count) for row, count in zip(*np.unique(train_actions, axis=0, return_counts=True))),
            key=lambda item: item[1],
        )[0],
        dtype=np.float32,
    )
    majority_pred = np.repeat(majority_action[None, :], len(val_actions), axis=0)
    return {
        "mean_regressor": compute_metrics(mean_pred, val_actions),
        "majority_action": compute_metrics(majority_pred, val_actions),
    }


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def default_metrics_path(model_path: Path) -> Path:
    return model_path.with_suffix(".metrics.json")


def format_metric(metrics: dict[str, object], key: str) -> str:
    value = metrics.get(key)
    return "n/a" if value is None else f"{float(value):.3f}"


def train(args: argparse.Namespace) -> None:
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio doit etre strictement entre 0 et 1.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.expand_ray_features and args.agent_id is None:
        raise ValueError("--expand-ray-features requiert --agent-id (nbRay depend de l'agent).")
    if args.add_ray_velocity and args.agent_id is None:
        raise ValueError("--add-ray-velocity requiert --agent-id (nbRay depend de l'agent).")

    sessions = load_sessions(
        args.data,
        agent_id=args.agent_id,
        input_source=args.input_source,
        expand_ray_features=args.expand_ray_features,
        add_ray_velocity=args.add_ray_velocity,
        agents_config=args.agents_config,
    )
    split = build_split(
        sessions,
        split=args.split,
        val_ratio=args.val_ratio,
        val_files=args.val_files,
        seed=args.seed,
    )
    train_weights = split.train_weights if args.use_sample_weight else np.ones_like(split.train_weights)
    val_weights = split.val_weights if args.use_sample_weight else np.ones_like(split.val_weights)

    if args.extreme_weight != 1.0:
        extreme_mask = (split.train_actions[:, 0] < args.extreme_throttle_below) | (
            np.abs(split.train_actions[:, 1]) > args.extreme_steer_above
        )
        train_weights = train_weights.copy()
        train_weights[extreme_mask] *= args.extreme_weight
        print(
            f"Extreme-weight: {int(extreme_mask.sum())}/{len(extreme_mask)} lignes "
            f"(throttle<{args.extreme_throttle_below} ou |steer|>{args.extreme_steer_above}) "
            f"x{args.extreme_weight}"
        )

    obs_normalization = None if args.no_normalize_obs else build_obs_normalization(split.train_obs)
    train_obs = apply_obs_normalization(split.train_obs, obs_normalization)
    val_obs = apply_obs_normalization(split.val_obs, obs_normalization)

    train_loader = build_loader(train_obs, split.train_actions, train_weights, args.batch_size, shuffle=True, seed=args.seed)
    val_loader = build_loader(val_obs, split.val_actions, val_weights, args.batch_size, shuffle=False, seed=args.seed)

    model = MLP(
        input_dim=train_obs.shape[1],
        output_dim=split.train_actions.shape[1],
        hidden=args.hidden_size,
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val_rmse = float("inf")
    best_val_metrics: dict[str, object] | None = None
    history: list[dict[str, float]] = []

    print(
        f"Split={split.split_name} | train={len(split.train_obs)} | val={len(split.val_obs)} | "
        f"hidden={args.hidden_size} | params={count_parameters(model)} | "
        f"normalize_obs={obs_normalization is not None}"
    )
    if split.split_name == "file":
        print(f"Validation files: {', '.join(split.val_files)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_weighted_loss = 0.0
        total_weight = 0.0

        for xb, yb, wb in train_loader:
            opt.zero_grad()
            pred = model(xb)
            loss = weighted_mse(pred, yb, wb)
            loss.backward()
            opt.step()
            batch_weight = wb.sum().item()
            total_weighted_loss += loss.item() * batch_weight
            total_weight += batch_weight

        train_loss = total_weighted_loss / max(1e-12, total_weight)
        val_metrics = evaluate_model(model, val_loader)
        val_rmse = float(val_metrics["rmse"])
        history.append(
            {
                "epoch": epoch,
                "train_loss_mse": float(train_loss),
                "val_mae": float(val_metrics["mae"]),
                "val_rmse": val_rmse,
                "val_exact_action_accuracy": val_metrics["exact_action_accuracy"],
                "val_action_within_0.25": float(val_metrics["action_within_0.25"]),
            }
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_val_metrics = val_metrics

        print(
            f"Epoch {epoch:03d} | train MSE {train_loss:.5f} | val MAE {float(val_metrics['mae']):.5f} | "
            f"val RMSE {val_rmse:.5f} | val exact {format_metric(val_metrics, 'exact_action_accuracy')} | "
            f"val within .25 {format_metric(val_metrics, 'action_within_0.25')}"
        )

    model.load_state_dict(best_state)
    train_metrics = evaluate_model(model, train_loader)
    val_metrics = evaluate_model(model, val_loader)
    baselines = build_baseline_metrics(split.train_actions, split.val_actions)

    metrics_summary = {
        "split": {
            "strategy": split.split_name,
            "val_ratio": args.val_ratio,
            "train_rows": int(len(split.train_obs)),
            "val_rows": int(len(split.val_obs)),
            "train_files": split.train_files,
            "val_files": split.val_files,
        },
        "model": {
            "type": "MLP",
            "hidden_size": args.hidden_size,
            "input_dim": int(split.train_obs.shape[1]),
            "output_dim": int(split.train_actions.shape[1]),
            "parameter_count": count_parameters(model),
            "best_epoch": best_epoch,
            "action_order": ACTION_ORDER,
            "obs_normalization": (
                {
                    "enabled": True,
                    "mean": [float(value) for value in obs_normalization.mean],
                    "std": [float(value) for value in obs_normalization.std],
                }
                if obs_normalization is not None
                else {"enabled": False}
            ),
            "ray_hit_feature": (
                {
                    "enabled": True,
                    "n_rays": load_agent_n_rays(args.agents_config, args.agent_id),
                    "no_hit_distance": NO_HIT_DISTANCE,
                }
                if args.expand_ray_features
                else {"enabled": False}
            ),
            "ray_velocity_feature": (
                {"enabled": True, "n_rays": load_agent_n_rays(args.agents_config, args.agent_id)}
                if args.add_ray_velocity
                else {"enabled": False}
            ),
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "extreme_weight": args.extreme_weight,
            "extreme_throttle_below": args.extreme_throttle_below,
            "extreme_steer_above": args.extreme_steer_above,
            "seed": args.seed,
            "normalize_obs": obs_normalization is not None,
        },
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "baselines": baselines,
        "history": history,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)

    metrics_path = args.metrics_out or default_metrics_path(args.out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics_summary, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"Best epoch: {best_epoch}")
    print(f"Validation MAE: {float(val_metrics['mae']):.5f}")
    print(f"Validation RMSE: {float(val_metrics['rmse']):.5f}")
    print(f"Validation exact action accuracy: {format_metric(val_metrics, 'exact_action_accuracy')}")
    print(f"Validation action within 0.10: {format_metric(val_metrics, 'action_within_0.10')}")
    print(f"Validation action within 0.25: {format_metric(val_metrics, 'action_within_0.25')}")
    print(f"Baseline mean RMSE: {float(baselines['mean_regressor']['rmse']):.5f}")
    print(f"Baseline majority exact action accuracy: {format_metric(baselines['majority_action'], 'exact_action_accuracy')}")
    if float(val_metrics["rmse"]) > float(baselines["mean_regressor"]["rmse"]):
        print("WARNING: le modele a une RMSE validation pire que la baseline moyenne.")
    if (
        val_metrics["exact_action_accuracy"] is not None
        and baselines["majority_action"]["exact_action_accuracy"] is not None
        and float(val_metrics["exact_action_accuracy"]) < float(baselines["majority_action"]["exact_action_accuracy"])
    ):
        print("WARNING: le modele a une exact action accuracy pire que la baseline majoritaire.")
    if float(val_metrics["action_within_0.25"]) < float(baselines["mean_regressor"]["action_within_0.25"]):
        print("WARNING: le modele est moins souvent a moins de 0.25 de l'action cible que la baseline moyenne.")
    print(f"Modele sauvegarde dans {args.out}")
    print(f"Metriques sauvegardees dans {metrics_path}")

    if best_val_metrics is None:
        raise RuntimeError("Aucune metrique de validation n'a ete capturee.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrainement supervise avec evaluation")
    parser.add_argument("data", type=Path, nargs="+", help="CSV genere par client.py (un ou plusieurs)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0, help="L2 (Adam weight_decay), 0 = desactive.")
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE, help="Taille cachee du petit MLP")
    parser.add_argument("--out", type=Path, default=Path("models/steering_mlp.pt"))
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="JSON ou sauvegarder les metriques (defaut: meme nom que le modele avec .metrics.json)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed pour reproductibilite")
    parser.add_argument(
        "--no-normalize-obs",
        action="store_true",
        help="Desactiver la normalisation des observations (deconseille).",
    )
    parser.add_argument(
        "--agent-id",
        type=int,
        default=None,
        help="Filtrer le dataset sur un agent_id precis (tel qu'enregistre dans le CSV).",
    )
    parser.add_argument(
        "--input-source",
        choices=["all", "clavier", "manette"],
        default="all",
        help="Filtrer sur la source d'entree (colonne input_source des CSV nettoyes par clean_dataset.py).",
    )
    parser.add_argument(
        "--use-sample-weight",
        action="store_true",
        help="Ponderer la loss avec la colonne sample_weight (CSV nettoyes par clean_dataset.py) pour rebalancer les actions.",
    )
    parser.add_argument(
        "--extreme-weight",
        type=float,
        default=1.0,
        help=(
            "Multiplie le poids des lignes a action extreme (freinage fort ou virage serre) dans la loss "
            "d'entrainement, pour empecher la regression MSE de les lisser vers une moyenne molle. "
            "1.0 = desactive."
        ),
    )
    parser.add_argument(
        "--extreme-throttle-below",
        type=float,
        default=0.1,
        help="Seuil de throttle en-dessous duquel une ligne est consideree comme un freinage fort.",
    )
    parser.add_argument(
        "--extreme-steer-above",
        type=float,
        default=0.7,
        help="Seuil de |steer| au-dessus duquel une ligne est consideree comme un virage serre.",
    )
    parser.add_argument(
        "--expand-ray-features",
        action="store_true",
        help=(
            "Separer chaque distance raycast en [distance plafonnee, indicateur rien-detecte] "
            "au lieu de la distance brute. Utile pour l'agent dont le capteur renvoie une valeur "
            "plafond bruitee (~256) quand rien n'est detecte. Requiert --agent-id."
        ),
    )
    parser.add_argument(
        "--add-ray-velocity",
        action="store_true",
        help=(
            "Ajoute la variation de chaque distance raycast entre deux pas consecutifs "
            "(vitesse de rapprochement), pour donner au MLP sans memoire un signal d'anticipation "
            "qu'une seule frame ne peut pas porter. Calcule une fois, dans l'ordre chronologique de "
            "chaque session, avant tout split/shuffle ulterieur. Requiert --agent-id."
        ),
    )
    parser.add_argument(
        "--agents-config",
        type=Path,
        default=DEFAULT_AGENTS_CONFIG,
        help="agents.json pour recuperer nbRay par agent (utilise par --expand-ray-features).",
    )
    parser.add_argument(
        "--split",
        choices=["file", "random"],
        default="file",
        help="file = validation par session CSV, random = split aleatoire ligne par ligne",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Part reservee a la validation",
    )
    parser.add_argument(
        "--val-files",
        type=Path,
        nargs="*",
        default=(),
        help="CSV explicitement reserves a la validation (utilisable avec --split file)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
