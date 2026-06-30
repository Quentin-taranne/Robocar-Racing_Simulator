#!/usr/bin/env python3
"""Visualise ce que predit un modele entraine (throttle/steer), pour comprendre
ses biais sans avoir a relancer le simulateur.

Genere deux figures :
  1. predicted_vs_curve.png : throttle predit en fonction de l'intensite du
     virage reellement enregistre (|steer|) -- montre directement si le modele
     module sa vitesse selon le virage ou s'il reste plat.
  2. timeline_<fichier>.png : throttle/steer predits vs enregistres au fil du
     temps sur une session, pour voir les ecarts en contexte.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "robocar_client"))
from train_model import MLP, load_sessions  # noqa: E402


def load_model_and_predict(model_path: Path, sessions) -> tuple[np.ndarray, np.ndarray]:
    import json

    metadata = json.loads(model_path.with_suffix(".metrics.json").read_text(encoding="utf-8"))
    norm = metadata["model"]["obs_normalization"]
    mean = np.array(norm["mean"], dtype=np.float32)
    std = np.array(norm["std"], dtype=np.float32)

    obs = np.concatenate([s.obs for s in sessions])
    actions = np.concatenate([s.actions for s in sessions])
    obs_n = (obs - mean) / std

    model = MLP(input_dim=obs.shape[1], output_dim=2, hidden=metadata["model"]["hidden_size"])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(obs_n, dtype=torch.float32)).numpy()
    return pred, actions


def plot_throttle_vs_curve(pred: np.ndarray, actions: np.ndarray, out_path: Path) -> None:
    abs_steer = np.abs(actions[:, 1])
    bins = np.linspace(0, 1, 11)
    bin_idx = np.digitize(abs_steer, bins) - 1
    bin_idx = np.clip(bin_idx, 0, len(bins) - 2)

    pred_means, actual_means, centers = [], [], []
    for b in range(len(bins) - 1):
        mask = bin_idx == b
        if mask.sum() < 5:
            continue
        pred_means.append(pred[mask, 0].mean())
        actual_means.append(actions[mask, 0].mean())
        centers.append((bins[b] + bins[b + 1]) / 2)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(centers, pred_means, marker="o", label="throttle predit par le modele")
    ax.plot(centers, actual_means, marker="s", linestyle="--", label="throttle enregistre (humain)")
    ax.set_xlabel("|steer| enregistre (intensite du virage)")
    ax.set_ylabel("throttle moyen")
    ax.set_ylim(0, 1.05)
    ax.set_title("Le modele module-t-il sa vitesse selon le virage ?")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_timeline(pred: np.ndarray, actions: np.ndarray, out_path: Path) -> None:
    n = len(pred)
    steps = np.arange(n)

    fig, (ax_t, ax_s) = plt.subplots(2, 1, sharex=True, figsize=(13, 6))
    ax_t.plot(steps, actions[:, 0], color="gray", linewidth=0.8, label="throttle enregistre")
    ax_t.plot(steps, pred[:, 0], color="tab:red", linewidth=0.8, label="throttle predit", alpha=0.8)
    ax_t.set_ylabel("throttle")
    ax_t.set_ylim(-0.05, 1.05)
    ax_t.legend(loc="upper right")

    ax_s.plot(steps, actions[:, 1], color="gray", linewidth=0.8, label="steer enregistre")
    ax_s.plot(steps, pred[:, 1], color="tab:blue", linewidth=0.8, label="steer predit", alpha=0.8)
    ax_s.set_ylabel("steer")
    ax_s.set_xlabel("frame")
    ax_s.set_ylim(-1.05, 1.05)
    ax_s.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise les predictions throttle/steer d'un modele")
    parser.add_argument("model", type=Path, help="Chemin du .pt (le .metrics.json doit exister a cote)")
    parser.add_argument("data", type=Path, nargs="+", help="CSV nettoyes (data_clean/ ou validation_files_clean/)")
    parser.add_argument("--agent-id", type=int, required=True)
    parser.add_argument("--input-source", choices=["all", "clavier", "manette"], default="manette")
    parser.add_argument("--expand-ray-features", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/predictions"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sessions = load_sessions(
        args.data,
        agent_id=args.agent_id,
        input_source=args.input_source,
        expand_ray_features=args.expand_ray_features,
    )
    pred, actions = load_model_and_predict(args.model, sessions)

    curve_path = args.out_dir / f"{args.model.stem}_throttle_vs_curve.png"
    plot_throttle_vs_curve(pred, actions, curve_path)
    print(f"Ecrit: {curve_path}")

    offset = 0
    for session in sessions:
        n = len(session.obs)
        session_pred = pred[offset : offset + n]
        session_actions = actions[offset : offset + n]
        offset += n
        timeline_path = args.out_dir / f"{args.model.stem}_timeline_{session.path.stem}.png"
        plot_timeline(session_pred, session_actions, timeline_path)
        print(f"Ecrit: {timeline_path}")


if __name__ == "__main__":
    main()
