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


HIDDEN_SIZE = 256
ACTION_ORDER = "throttle-steer"  # ordre impose : col0 throttle, col1 steer


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


@dataclass
class DatasetSplit:
    train_obs: np.ndarray
    train_actions: np.ndarray
    val_obs: np.ndarray
    val_actions: np.ndarray
    train_files: list[str]
    val_files: list[str]
    split_name: str


def reorder_actions(actions: np.ndarray) -> np.ndarray:
    if actions.shape[1] >= 2 and ACTION_ORDER == "throttle-steer":
        c0_min, c0_max = actions[:, 0].min(), actions[:, 0].max()
        c1_min, c1_max = actions[:, 1].min(), actions[:, 1].max()
        if c0_min < -0.5 and c0_max <= 1.0 and c1_min >= -0.1 and c1_max > 0.5:
            actions = actions[:, [1, 0]]
    return actions


def load_sessions(paths: Sequence[Path], agent_id: int | None = None) -> list[SessionData]:
    sessions: list[SessionData] = []
    for path in paths:
        df = pd.read_csv(path)
        if agent_id is not None:
            df = df[df["agent_id"] == agent_id]
        if df.empty:
            continue
        obs = np.stack(df["obs"].apply(json.loads).values).astype(np.float32)
        actions = np.stack(df["action"].apply(json.loads).values).astype(np.float32)
        actions = reorder_actions(actions)
        sessions.append(SessionData(path=path, obs=obs, actions=actions))
    if not sessions:
        raise ValueError("Aucun echantillon apres filtrage (paths/agent_id).")
    return sessions


def stack_sessions(sessions: Sequence[SessionData]) -> tuple[np.ndarray, np.ndarray]:
    obs = np.concatenate([session.obs for session in sessions], axis=0)
    actions = np.concatenate([session.actions for session in sessions], axis=0)
    return obs.astype(np.float32), actions.astype(np.float32)


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

    train_obs, train_actions = stack_sessions(train_sessions)
    val_obs, val_actions = stack_sessions(val_sessions)
    return DatasetSplit(
        train_obs=train_obs,
        train_actions=train_actions,
        val_obs=val_obs,
        val_actions=val_actions,
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
    obs, actions = stack_sessions(sessions)
    dataset = TensorDataset(
        torch.tensor(obs, dtype=torch.float32),
        torch.tensor(actions, dtype=torch.float32),
    )
    train_len = int(len(dataset) * (1.0 - val_ratio))
    train_len = min(max(1, train_len), len(dataset) - 1)
    val_len = len(dataset) - train_len
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=generator)

    train_obs = train_ds.dataset.tensors[0][train_ds.indices].cpu().numpy()
    train_actions = train_ds.dataset.tensors[1][train_ds.indices].cpu().numpy()
    val_obs = val_ds.dataset.tensors[0][val_ds.indices].cpu().numpy()
    val_actions = val_ds.dataset.tensors[1][val_ds.indices].cpu().numpy()

    file_names = [session.path.name for session in sessions]
    return DatasetSplit(
        train_obs=train_obs,
        train_actions=train_actions,
        val_obs=val_obs,
        val_actions=val_actions,
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


def build_loader(obs: np.ndarray, actions: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(obs, dtype=torch.float32),
        torch.tensor(actions, dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


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
        metrics["throttle_accuracy"] = float(np.mean(snapped[:, 0] == target[:, 0]))
    if target.shape[1] >= 2:
        metrics["steer_accuracy"] = float(np.mean(snapped[:, 1] == target[:, 1]))
    if target.shape[1] >= 2:
        metrics["exact_action_accuracy"] = float(np.mean(np.all(snapped[:, :2] == target[:, :2], axis=1)))
    else:
        metrics["exact_action_accuracy"] = float(np.mean(np.all(snapped == target, axis=1)))

    return metrics


def collect_predictions(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> tuple[np.ndarray, np.ndarray, float]:
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    total_loss = 0.0
    total_rows = 0

    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb)
            batch_rows = len(xb)
            total_loss += loss_fn(pred, yb).item() * batch_rows
            total_rows += batch_rows
            preds.append(pred.cpu().numpy())
            targets.append(yb.cpu().numpy())

    pred_np = np.concatenate(preds, axis=0)
    target_np = np.concatenate(targets, axis=0)
    return pred_np, target_np, total_loss / max(1, total_rows)


def evaluate_model(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> dict[str, object]:
    pred, target, loss = collect_predictions(model, loader, loss_fn)
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


def train(args: argparse.Namespace) -> None:
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio doit etre strictement entre 0 et 1.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    sessions = load_sessions(args.data, agent_id=args.agent_id)
    split = build_split(
        sessions,
        split=args.split,
        val_ratio=args.val_ratio,
        val_files=args.val_files,
        seed=args.seed,
    )

    train_loader = build_loader(split.train_obs, split.train_actions, args.batch_size, shuffle=True, seed=args.seed)
    val_loader = build_loader(split.val_obs, split.val_actions, args.batch_size, shuffle=False, seed=args.seed)

    model = MLP(
        input_dim=split.train_obs.shape[1],
        output_dim=split.train_actions.shape[1],
        hidden=args.hidden_size,
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val_rmse = float("inf")
    best_val_metrics: dict[str, object] | None = None
    history: list[dict[str, float]] = []

    print(
        f"Split={split.split_name} | train={len(split.train_obs)} | val={len(split.val_obs)} | "
        f"hidden={args.hidden_size} | params={count_parameters(model)}"
    )
    if split.split_name == "file":
        print(f"Validation files: {', '.join(split.val_files)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0

        for xb, yb in train_loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            batch_rows = len(xb)
            total_loss += loss.item() * batch_rows
            total_rows += batch_rows

        train_loss = total_loss / max(1, total_rows)
        val_metrics = evaluate_model(model, val_loader, loss_fn)
        val_rmse = float(val_metrics["rmse"])
        history.append(
            {
                "epoch": epoch,
                "train_loss_mse": float(train_loss),
                "val_mae": float(val_metrics["mae"]),
                "val_rmse": val_rmse,
                "val_exact_action_accuracy": float(val_metrics["exact_action_accuracy"]),
            }
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_val_metrics = val_metrics

        print(
            f"Epoch {epoch:03d} | train MSE {train_loss:.5f} | val MAE {float(val_metrics['mae']):.5f} | "
            f"val RMSE {val_rmse:.5f} | val exact {float(val_metrics['exact_action_accuracy']):.3f}"
        )

    model.load_state_dict(best_state)
    train_metrics = evaluate_model(model, train_loader, loss_fn)
    val_metrics = evaluate_model(model, val_loader, loss_fn)
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
    print(f"Validation exact action accuracy: {float(val_metrics['exact_action_accuracy']):.3f}")
    print(f"Baseline mean RMSE: {float(baselines['mean_regressor']['rmse']):.5f}")
    print(f"Baseline majority exact action accuracy: {float(baselines['majority_action']['exact_action_accuracy']):.3f}")
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
        "--agent-id",
        type=int,
        default=None,
        help="Filtrer le dataset sur un agent_id precis (tel qu'enregistre dans le CSV).",
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
