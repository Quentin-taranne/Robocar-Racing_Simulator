"""Entraînement supervisé minimal sur les traces clavier."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


HIDDEN_SIZE = 256
ACTION_ORDER = "throttle-steer"  # ordre imposé : col0 throttle, col1 steer


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int = HIDDEN_SIZE) -> None:
        super().__init__() 
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
            nn.Tanh(),  # actions sont bornées entre -1 et 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_dataset(paths: Sequence[Path], agent_id: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    obs_list = []
    act_list = []
    for path in paths:
        df = pd.read_csv(path)
        if agent_id is not None:
            df = df[df["agent_id"] == agent_id]
        if df.empty:
            continue
        obs_list.append(np.stack(df["obs"].apply(json.loads).values))
        actions = np.stack(df["action"].apply(json.loads).values)
        # Réordonne si nécessaire (imposé throttle-steer)
        if actions.shape[1] >= 2 and ACTION_ORDER == "throttle-steer":
            # Les données de collecte sont déjà throttle-steer ; si jamais elles étaient inversées, les mettre en ordre
            # Detect simple cas: si col0 semble être steer dans [-1,1] et col1 dans [0,1], on swap
            c0_min, c0_max = actions[:, 0].min(), actions[:, 0].max()
            c1_min, c1_max = actions[:, 1].min(), actions[:, 1].max()
            if c0_min < -0.5 and c0_max <= 1.0 and c1_min >= -0.1 and c1_max > 0.5:
                actions = actions[:, [1, 0]]
        act_list.append(actions)
    if not obs_list:
        raise ValueError("Aucun échantillon après filtrage (paths/agent_id).")
    obs = np.concatenate(obs_list, axis=0)
    actions = np.concatenate(act_list, axis=0)
    return obs.astype(np.float32), actions.astype(np.float32)


def train(args: argparse.Namespace) -> None:
    # Fix seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    obs, actions = load_dataset(args.data, agent_id=args.agent_id)
    dataset = TensorDataset(
        torch.tensor(obs, dtype=torch.float32),
        torch.tensor(actions, dtype=torch.float32),
    )

    train_len = int(len(dataset) * 0.8)
    val_len = max(1, len(dataset) - train_len)
    gen = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, generator=gen)

    model = MLP(input_dim=obs.shape[1], output_dim=actions.shape[1], hidden=HIDDEN_SIZE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            opt.zero_grad()
            pred = model(xb)
            loss = mse(pred, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
        avg_loss = total_loss / len(train_ds)

        model.eval()
        with torch.no_grad():
            abs_err, count = 0.0, 0
            for xb, yb in val_loader:
                pred = model(xb)
                abs_err += torch.abs(pred - yb).sum().item()
                count += pred.numel()
            mae = abs_err / count if count else 0.0
        print(f"Epoch {epoch:03d} | train MSE {avg_loss:.5f} | val MAE {mae:.5f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"Modèle sauvegardé dans {args.out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Petit entraînement supervisé")
    parser.add_argument("data", type=Path, nargs="+", help="CSV généré par client.py (un ou plusieurs)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=Path, default=Path("models/steering_mlp.pt"))
    parser.add_argument("--seed", type=int, default=42, help="Seed pour reproductibilité")
    parser.add_argument(
        "--agent-id",
        type=int,
        default=None,
        help="Filtrer le dataset sur un agent_id précis (tel qu'enregistré dans le CSV).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
