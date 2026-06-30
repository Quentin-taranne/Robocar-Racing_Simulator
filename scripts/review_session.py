#!/usr/bin/env python3
"""Revue visuelle manuelle des sessions de conduite, frame par frame, pour marquer
les lignes a exclure de l'entrainement (crash, blocage, virage rate...).

Chaque ligne du CSV est une frame. La fenetre affiche :
  - en haut, un apercu throttle/steer au fil du temps avec un curseur vertical
    indiquant la frame courante, et les plages deja marquees "supprime" en rouge.
  - en bas, une reconstruction en temps reel des raycasts de la frame courante
    (vue du dessus : la voiture au centre, chaque rayon trace dans sa direction
    avec sa distance), pour se reperer spatialement (mur proche, virage qui
    s'ouvre, etc).

Navigation :
  fleche droite / gauche   avancer / reculer d'une frame (appui maintenu = defile)
  d (maintenue) + fleche   les frames visitees en la maintenant sont marquees
                           "supprime" (rouge). Relacher 'd' repasse en mode
                           navigation simple (les frames visitees ne sont plus
                           marquees).
  u                        demarque la frame courante (la remet "gardee")
  c                        efface toutes les marques du fichier courant
  s                        sauvegarde les marques du fichier courant, passe au suivant
  n                        passe au fichier suivant sans sauvegarder
  q                        quitte (marques non sauvegardees du fichier courant perdues)

Les marques sont fusionnees en plages [debut, fin] (timestamp absolu) et stockees
dans un JSON (`data_clean_exclusions.json` par defaut), relu par `clean_dataset.py`
pour retirer ces lignes avant tout le reste du nettoyage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("MacOSX" if sys.platform == "darwin" else "TkAgg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean_dataset import load_raw  # noqa: E402

DEFAULT_AGENTS_CONFIG = Path("robocar_client/agents.json")
DEFAULT_FOV = 180.0


def load_exclusions(path: Path) -> dict[str, list[list[float]]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_exclusions(path: Path, exclusions: dict[str, list[list[float]]]) -> None:
    path.write_text(json.dumps(exclusions, indent=2, ensure_ascii=True), encoding="utf-8")


def get_agent_fov(agent_id: int, agents_config: Path) -> float:
    if agents_config.exists():
        try:
            payload = json.loads(agents_config.read_text(encoding="utf-8"))
            agents = payload.get("agents", [])
            if 0 <= agent_id < len(agents):
                fov = agents[agent_id].get("fov")
                if fov:
                    return float(fov)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
    return DEFAULT_FOV


def mask_to_ranges(mask: np.ndarray, timestamps: np.ndarray) -> list[list[float]]:
    ranges: list[list[float]] = []
    n = len(mask)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        ranges.append([float(timestamps[i]), float(timestamps[j])])
        i = j + 1
    return ranges


def ranges_to_mask(ranges: list[list[float]], timestamps: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(timestamps), dtype=bool)
    for start, end in ranges:
        mask |= (timestamps >= start) & (timestamps <= end)
    return mask


def review_file(path: Path, exclusions: dict[str, list[list[float]]], agents_config: Path) -> bool:
    """Affiche une session frame par frame. Renvoie False si l'utilisateur quitte."""
    df = load_raw(path)
    if df.empty:
        print(f"[SKIP] {path.name} est vide.")
        return True

    timestamps = df["timestamp"].to_numpy(dtype=float)
    t0 = timestamps[0]
    elapsed = timestamps - t0
    throttle = df["throttle"].to_numpy()
    steer = df["steer"].to_numpy()
    agent_id = int(df["agent_id"].iloc[0])
    n = len(df)

    obs_arr = np.stack(df["obs_list"].to_numpy())
    active_cols = np.where(np.any(obs_arr != -1.0, axis=0))[0]
    rays = obs_arr[:, active_cols] if len(active_cols) else obs_arr
    n_rays = rays.shape[1]
    fov = get_agent_fov(agent_id, agents_config)
    angles_rad = np.radians(np.linspace(-fov / 2.0, fov / 2.0, n_rays))
    max_distance = float(rays.max()) if rays.size else 1.0

    deleted = ranges_to_mask(exclusions.get(path.name, []), timestamps)

    fig = plt.figure(figsize=(11, 9))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 4])
    ax_overview_t = fig.add_subplot(gs[0])
    ax_overview_s = fig.add_subplot(gs[1], sharex=ax_overview_t)
    ax_fan = fig.add_subplot(gs[2])

    ax_overview_t.plot(elapsed, throttle, color="tab:blue", linewidth=0.7)
    ax_overview_t.set_ylabel("throttle")
    ax_overview_t.set_ylim(-1.05, 1.05)
    ax_overview_s.plot(elapsed, steer, color="tab:green", linewidth=0.7)
    ax_overview_s.set_ylabel("steer")
    ax_overview_s.set_xlabel("secondes depuis le debut de la session")
    ax_overview_s.set_ylim(-1.05, 1.05)

    cursor_lines = [ax_overview_t.axvline(0, color="black", linewidth=1.2, zorder=5) for _ in range(1)]
    cursor_lines.append(ax_overview_s.axvline(0, color="black", linewidth=1.2, zorder=5))

    deleted_patches: list = []

    def redraw_deleted_spans() -> None:
        for patch in deleted_patches:
            patch.remove()
        deleted_patches.clear()
        for start, end in mask_to_ranges(deleted, timestamps):
            for ax in (ax_overview_t, ax_overview_s):
                deleted_patches.append(ax.axvspan(start - t0, end - t0, color="red", alpha=0.3, zorder=1))
        fig.canvas.draw_idle()

    redraw_deleted_spans()

    margin = max_distance * 1.1 if max_distance > 0 else 1.0
    ax_fan.set_xlim(-margin, margin)
    ax_fan.set_ylim(-margin * 0.2, margin)
    ax_fan.set_aspect("equal")
    ax_fan.set_xticks([])
    ax_fan.set_yticks([])
    ax_fan.scatter([0], [0], marker="^", s=220, color="black", zorder=5)
    ray_lines = LineCollection([], linewidths=1.2, cmap="RdYlGn", zorder=3)
    ray_lines.set_clim(0, max_distance)
    ax_fan.add_collection(ray_lines)
    ray_tips = ax_fan.scatter([], [], s=18, c=[], cmap="RdYlGn", vmin=0, vmax=max_distance, zorder=4)
    fan_title = ax_fan.set_title("")

    state = {"idx": 0, "held_delete": False, "quit": False}

    def render_fan(idx: int) -> None:
        distances = rays[idx]
        x = distances * np.sin(angles_rad)
        y = distances * np.cos(angles_rad)
        segments = [[(0.0, 0.0), (float(xi), float(yi))] for xi, yi in zip(x, y)]
        ray_lines.set_segments(segments)
        ray_lines.set_array(distances)
        ray_tips.set_offsets(np.column_stack([x, y]))
        ray_tips.set_array(distances)
        status = "SUPPRIME" if deleted[idx] else "garde"
        fan_title.set_text(
            f"{path.name}  |  frame {idx + 1}/{n}  |  t={elapsed[idx]:.2f}s  |  "
            f"throttle={throttle[idx]:+.2f} steer={steer[idx]:+.2f}  |  [{status}]"
        )
        fan_title.set_color("tab:red" if deleted[idx] else "black")
        for line in cursor_lines:
            line.set_xdata([elapsed[idx], elapsed[idx]])
        fig.canvas.draw_idle()

    render_fan(state["idx"])

    def move(delta: int) -> None:
        new_idx = int(np.clip(state["idx"] + delta, 0, n - 1))
        if new_idx == state["idx"]:
            return
        state["idx"] = new_idx
        if state["held_delete"]:
            deleted[new_idx] = True
            redraw_deleted_spans()
        render_fan(new_idx)

    def on_key_press(event) -> None:
        if event.key == "right":
            move(1)
        elif event.key == "left":
            move(-1)
        elif event.key == "d":
            state["held_delete"] = True
            deleted[state["idx"]] = True
            redraw_deleted_spans()
            render_fan(state["idx"])
        elif event.key == "u":
            deleted[state["idx"]] = False
            redraw_deleted_spans()
            render_fan(state["idx"])
        elif event.key == "c":
            deleted[:] = False
            redraw_deleted_spans()
            render_fan(state["idx"])
            print("  toutes les marques effacees pour ce fichier.")
        elif event.key == "s":
            exclusions[path.name] = mask_to_ranges(deleted, timestamps)
            print(f"  sauvegarde de {int(deleted.sum())} ligne(s) supprimee(s) pour {path.name}")
            plt.close(fig)
        elif event.key == "n":
            print(f"  {path.name} passe sans sauvegarder.")
            plt.close(fig)
        elif event.key == "q":
            state["quit"] = True
            plt.close(fig)

    def on_key_release(event) -> None:
        if event.key == "d":
            state["held_delete"] = False

    fig.canvas.mpl_connect("key_press_event", on_key_press)
    fig.canvas.mpl_connect("key_release_event", on_key_release)
    plt.tight_layout()
    plt.show()

    return not state["quit"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Revue visuelle frame par frame des sessions Robocar")
    parser.add_argument("files", type=Path, nargs="+", help="CSV bruts a revoir (data/drive_*.csv)")
    parser.add_argument(
        "--exclusions",
        type=Path,
        default=Path("data_clean_exclusions.json"),
        help="Fichier JSON ou stocker les lignes exclues (relu par clean_dataset.py)",
    )
    parser.add_argument(
        "--agents-config",
        type=Path,
        default=DEFAULT_AGENTS_CONFIG,
        help="agents.json pour recuperer le fov par agent (sinon 180 par defaut)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exclusions = load_exclusions(args.exclusions)

    for path in sorted(args.files):
        print(f"=== {path.name} ===")
        keep_going = review_file(path, exclusions, args.agents_config)
        save_exclusions(args.exclusions, exclusions)
        if not keep_going:
            print("Arret demande.")
            break

    print(f"Exclusions sauvegardees dans {args.exclusions}")


if __name__ == "__main__":
    main()
