#!/usr/bin/env python3
"""Nettoyage des traces CSV Robocar avant entrainement.

Trois operations, par fichier (= une session = un agent_id) :
  1. Suppression des frames consecutives strictement identiques (obs ET action
     inchangees) : ne portent aucun signal d'apprentissage supplementaire.
  2. Resolution du bruit de label : quand une meme observation exacte est vue
     plusieurs fois dans la session avec des actions differentes, l'action est
     remplacee par la moyenne des actions observees pour cette observation
     (cible qui minimise l'erreur quadratique pour un etat ambigu).
  3. Classification clavier/manette par session (les valeurs clavier sont
     discretes a {-1,0,1}, celles de la manette sont continues) et calcul d'un
     `sample_weight` par ligne pour rebalancer la distribution globale
     d'actions (par agent) sans supprimer de donnees.
  4. Detection heuristique des segments "bloque" (observation quasi figee
     pendant une duree soutenue alors que le throttle reste actif) : un proxy
     pour les crashs/collisions, puisque `reward` et `done` ne portent aucun
     signal exploitable dans ce dataset. Ces segments sont listes dans le
     rapport pour validation ; ils ne sont retires des CSV nettoyes que si
     `--drop-stuck-segments` est passe explicitement.
  5. Exclusions manuelles : si un fichier JSON (voir `scripts/review_session.py`)
     liste des plages [debut, fin] par fichier (timestamps absolus), ces plages
     sont retirees en priorite, avant meme la detection heuristique. C'est la
     seule source de verite fiable pour les crashs que l'heuristique ne detecte
     pas (sortie de piste a vitesse normale, par exemple).
  6. Detection des "retour au depart" (agent 0 uniquement) : le simulateur
     reinitialise la voiture a une observation fixe apres un crash. On detecte
     ce saut brutal d'observation puis on cherche, dans les secondes qui
     precedent, un rayon tres proche (mur touche) ; si trouve, les frames entre
     ce contact et le reset sont la "decision ratee" qui a mene au crash et
     sont listees pour suppression. Valide uniquement sur le capteur 10-rayons
     d'agent 0 ; sur le capteur 36-rayons d'agent 1, une valeur sentinelle de
     "aucun obstacle detecte" cree trop de faux positifs (voir limites).

Limite connue : la moyenne en cas de conflit de label suppose que l'ambiguite
est du bruit autour d'une valeur continue. Si un meme etat appelle vraiment
deux comportements differents (information manquante dans l'observation), la
moyenne lisse ce signal au lieu de le capturer : c'est une limite du
clonage de comportement par regression, pas un bug du nettoyage. De meme, le
detecteur de blocage est une heuristique sur l'observation seule : il rate les
crashs qui ne figent pas les raycasts (sortie de piste a vitesse normale) et
peut, rarement, flaguer un arret legitime prolonge avec throttle relance. Le
detecteur de retour-au-depart n'est active que pour agent 0 : sur agent 1
(36 rayons, champ de vision etroit), une valeur sentinelle "aucun obstacle"
clignote sur les rayons en bordure pendant la conduite normale et produit des
sauts d'observation aussi grands que les vrais resets, sans qu'on ait trouve
de critere fiable pour les distinguer.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("timestamp", "episode", "step", "agent_id", "reward", "done", "action", "obs")

STEER_BINS = (-1.0001, -0.5, -0.05, 0.05, 0.5, 1.0001)
THROTTLE_BINS = (-0.0001, 0.05, 0.95, 1.0001)


@dataclass
class StuckSegment:
    file_name: str
    start_row: int
    end_row: int
    rows: int
    start_ts: float
    end_ts: float
    duration: float
    mean_throttle: float
    mean_abs_steer: float


@dataclass
class ResetCrashSegment:
    file_name: str
    start_row: int
    end_row: int
    rows: int
    start_ts: float
    end_ts: float
    duration: float
    min_ray_distance: float
    reset_row: int


@dataclass
class FileCleanStats:
    name: str
    agent_id: int
    input_source: str
    rows_before: int
    rows_after_dedup: int
    duplicates_dropped: int
    conflict_groups: int
    conflict_rows_resolved: int
    stuck_segments: int
    stuck_rows: int
    stuck_rows_dropped: int
    manual_excluded_rows: int
    reset_crash_segments: int
    reset_crash_rows: int
    reset_crash_rows_dropped: int


@dataclass
class CleanReport:
    files: list[FileCleanStats] = field(default_factory=list)
    rows_before_total: int = 0
    rows_after_total: int = 0
    stuck_segments: list[StuckSegment] = field(default_factory=list)
    reset_crash_segments: list[ResetCrashSegment] = field(default_factory=list)


def is_discrete_value(value: float) -> bool:
    return np.isclose(value, -1.0) or np.isclose(value, 0.0) or np.isclose(value, 1.0)


def classify_input_source(throttle: np.ndarray, steer: np.ndarray) -> str:
    def continuous_fraction(values: np.ndarray) -> float:
        discrete = np.isclose(values, -1.0) | np.isclose(values, 0.0) | np.isclose(values, 1.0)
        return 1.0 - float(discrete.mean()) if len(values) else 0.0

    cont_frac = max(continuous_fraction(throttle), continuous_fraction(steer))
    return "manette" if cont_frac > 0.01 else "clavier"


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{path}: colonnes manquantes {missing}")
    df["action_list"] = df["action"].apply(json.loads)
    df["obs_list"] = df["obs"].apply(json.loads)
    df["throttle"] = df["action_list"].apply(lambda a: float(a[0]))
    df["steer"] = df["action_list"].apply(lambda a: float(a[1]))
    return df


def load_manual_exclusions(path: Path) -> dict[str, list[tuple[float, float]]]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {name: [(float(a), float(b)) for a, b in ranges] for name, ranges in raw.items()}


def apply_manual_exclusions(df: pd.DataFrame, ranges: list[tuple[float, float]]) -> tuple[pd.DataFrame, int]:
    if not ranges or df.empty:
        return df, 0
    timestamps = df["timestamp"].to_numpy(dtype=float)
    exclude_mask = np.zeros(len(df), dtype=bool)
    for start, end in ranges:
        exclude_mask |= (timestamps >= start) & (timestamps <= end)
    kept = df.loc[~exclude_mask].reset_index(drop=True)
    return kept, int(exclude_mask.sum())


def drop_consecutive_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if len(df) < 2:
        return df, 0
    obs_arr = np.stack(df["obs_list"].to_numpy())
    same_obs = np.all(obs_arr[1:] == obs_arr[:-1], axis=1)
    throttle = df["throttle"].to_numpy()
    steer = df["steer"].to_numpy()
    same_action = (throttle[1:] == throttle[:-1]) & (steer[1:] == steer[:-1])
    drop_mask = np.concatenate([[False], same_obs & same_action])
    kept = df.loc[~drop_mask].reset_index(drop=True)
    return kept, int(drop_mask.sum())


STUCK_MAX_OBS_DELTA = 1.0  # variation max (en unite raycast) entre deux pas consecutifs pour etre "quasi figee"
STUCK_MIN_THROTTLE = 0.1  # throttle au-dessus duquel la voiture "essaie" d'avancer
STUCK_MIN_SECONDS = 1.0  # duree minimale du blocage pour etre signale


def find_stuck_segments(
    df: pd.DataFrame,
    *,
    max_obs_delta: float = STUCK_MAX_OBS_DELTA,
    min_throttle: float = STUCK_MIN_THROTTLE,
    min_seconds: float = STUCK_MIN_SECONDS,
) -> tuple[np.ndarray, list[dict]]:
    """Detecte les runs de pas consecutifs ou l'observation bouge a peine alors
    que le throttle est actif : signature typique d'une voiture bloquee contre
    un mur/obstacle. Calcule sur les donnees brutes (avant dedup) pour que la
    duree du blocage soit fidele aux timestamps reels.
    """
    n = len(df)
    if n < 2:
        return np.zeros(n, dtype=bool), []

    obs_arr = np.stack(df["obs_list"].to_numpy())
    deltas = np.max(np.abs(obs_arr[1:] - obs_arr[:-1]), axis=1)
    low_movement = np.concatenate([[False], deltas <= max_obs_delta])
    active_throttle = df["throttle"].to_numpy() > min_throttle
    candidate = low_movement & active_throttle

    timestamps = df["timestamp"].to_numpy(dtype=float)
    throttle = df["throttle"].to_numpy()
    steer = df["steer"].to_numpy()

    mask = np.zeros(n, dtype=bool)
    segments: list[dict] = []
    i = 0
    while i < n:
        if not candidate[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and candidate[j + 1]:
            j += 1
        duration = float(timestamps[j] - timestamps[i])
        if duration >= min_seconds:
            mask[i : j + 1] = True
            segments.append(
                {
                    "start_row": i,
                    "end_row": j,
                    "rows": j - i + 1,
                    "start_ts": float(timestamps[i]),
                    "end_ts": float(timestamps[j]),
                    "duration": duration,
                    "mean_throttle": float(throttle[i : j + 1].mean()),
                    "mean_abs_steer": float(np.abs(steer[i : j + 1]).mean()),
                }
            )
        i = j + 1

    return mask, segments


RESET_JUMP_THRESHOLD = 150.0  # saut euclidien d'observation entre 2 pas pour suspecter un teleport/reset
RESET_CLOSE_THRESHOLD = 30.0  # distance de rayon en-dessous de laquelle on suspecte un contact avec un obstacle
RESET_LOOKBACK_SECONDS = 3.0  # fenetre de recherche du contact, avant le reset
RESET_DETECTOR_AGENT_IDS = (0,)  # valide uniquement sur le capteur 10-rayons d'agent 0, cf. limites connues


def find_reset_crash_segments(
    df: pd.DataFrame,
    *,
    jump_threshold: float = RESET_JUMP_THRESHOLD,
    close_threshold: float = RESET_CLOSE_THRESHOLD,
    lookback_seconds: float = RESET_LOOKBACK_SECONDS,
) -> tuple[np.ndarray, list[dict]]:
    """Detecte les "retour au depart" (teleport apres crash) : un saut brutal et
    instantane de l'observation. Si un rayon tres proche est trouve dans les
    `lookback_seconds` precedant ce saut, les frames entre ce contact et le
    reset (exclu, puisque c'est deja la position de redemarrage valide) sont
    la decision de conduite qui a mene au crash.
    """
    n = len(df)
    if n < 2:
        return np.zeros(n, dtype=bool), []

    obs_arr = np.stack(df["obs_list"].to_numpy())
    active_cols = np.where(np.any(obs_arr != -1.0, axis=0))[0]
    rays = obs_arr[:, active_cols] if len(active_cols) else obs_arr

    deltas = np.linalg.norm(rays[1:] - rays[:-1], axis=1)
    reset_rows = np.where(deltas > jump_threshold)[0] + 1
    timestamps = df["timestamp"].to_numpy(dtype=float)

    mask = np.zeros(n, dtype=bool)
    segments: list[dict] = []
    for reset_row in reset_rows:
        lo = reset_row - 1
        while lo > 0 and (timestamps[reset_row] - timestamps[lo]) <= lookback_seconds:
            lo -= 1
        window = rays[lo:reset_row]
        if len(window) == 0:
            continue
        min_per_row = window.min(axis=1)
        close_local_idx = int(np.argmin(min_per_row))
        min_value = float(min_per_row[close_local_idx])
        if min_value > close_threshold:
            continue
        start_row = lo + close_local_idx
        end_row = reset_row - 1
        if end_row < start_row:
            continue
        mask[start_row : end_row + 1] = True
        segments.append(
            {
                "start_row": start_row,
                "end_row": end_row,
                "rows": end_row - start_row + 1,
                "start_ts": float(timestamps[start_row]),
                "end_ts": float(timestamps[end_row]),
                "duration": float(timestamps[end_row] - timestamps[start_row]),
                "min_ray_distance": min_value,
                "reset_row": int(reset_row),
            }
        )

    return mask, segments


NOISE_SPREAD_THRESHOLD = 0.3  # au-dela, le desaccord est traite comme un vrai choix divergent, pas du bruit


def resolve_label_conflicts(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Lisse les conflits de label qui ressemblent a du bruit (jitter clavier/manette
    autour d'une meme valeur), mais preserve les desaccords ou les actions divergent
    franchement (ex: -0.8 vs +0.7 pour la meme observation) : moyenner ce cas produirait
    un label "tout droit" alors que la voiture devait reellement tourner d'un cote ou
    de l'autre. Seuil arbitraire mais explicite : un ecart max-min <= 0.3 est considere
    comme du bruit, au-dela c'est une divergence de comportement a garder telle quelle.
    """
    if df.empty:
        return df, 0, 0
    df = df.copy()
    df["_obs_key"] = [tuple(obs) for obs in df["obs_list"]]
    group = df.groupby("_obs_key")[["throttle", "steer"]]
    means = group.transform("mean")
    spreads = group.transform(lambda s: s.max() - s.min())
    distinct_actions = group.transform(lambda s: s.nunique())

    conflict_groups_keys: set = set()
    smoothed_mask_total = pd.Series(False, index=df.index)
    for column in ("throttle", "steer"):
        has_conflict = distinct_actions[column] > 1
        conflict_groups_keys.update(df.loc[has_conflict, "_obs_key"].unique())
        smooth_mask = has_conflict & (spreads[column] <= NOISE_SPREAD_THRESHOLD)
        df.loc[smooth_mask, column] = means.loc[smooth_mask, column]
        smoothed_mask_total |= smooth_mask

    resolved_rows = int(smoothed_mask_total.sum())
    conflict_groups = len(conflict_groups_keys)
    df = df.drop(columns=["_obs_key"])
    return df, conflict_groups, resolved_rows


def compute_sample_weights(throttle: np.ndarray, steer: np.ndarray) -> np.ndarray:
    steer_bin = np.digitize(steer, STEER_BINS)
    throttle_bin = np.digitize(throttle, THROTTLE_BINS)
    combo = steer_bin * 10 + throttle_bin
    counts = pd.Series(combo).value_counts()
    freq = pd.Series(combo).map(counts).to_numpy().astype(np.float64)
    weight = 1.0 / freq
    weight = weight / weight.mean()
    return weight.astype(np.float32)


def clean_file(
    path: Path,
    *,
    drop_stuck_segments: bool,
    drop_reset_crashes: bool,
    manual_exclusions: dict[str, list[tuple[float, float]]] | None = None,
) -> tuple[pd.DataFrame, FileCleanStats, list[StuckSegment], list[ResetCrashSegment]]:
    df = load_raw(path)
    rows_before = len(df)
    agent_id = int(df["agent_id"].iloc[0]) if rows_before else -1
    input_source = classify_input_source(df["throttle"].to_numpy(), df["steer"].to_numpy())

    df, manual_excluded_rows = apply_manual_exclusions(df, (manual_exclusions or {}).get(path.name, []))

    stuck_mask, raw_segments = find_stuck_segments(df)
    df["stuck"] = stuck_mask
    segments = [StuckSegment(file_name=path.name, **segment) for segment in raw_segments]
    stuck_rows_total = int(stuck_mask.sum())

    if agent_id in RESET_DETECTOR_AGENT_IDS:
        reset_mask, raw_reset_segments = find_reset_crash_segments(df)
    else:
        reset_mask, raw_reset_segments = np.zeros(len(df), dtype=bool), []
    df["reset_crash"] = reset_mask
    reset_segments = [ResetCrashSegment(file_name=path.name, **segment) for segment in raw_reset_segments]
    reset_crash_rows_total = int(reset_mask.sum())

    df, dropped = drop_consecutive_duplicates(df)
    rows_after_dedup = len(df)
    df, conflict_groups, conflict_rows = resolve_label_conflicts(df)

    stuck_rows_dropped = 0
    if drop_stuck_segments:
        stuck_rows_dropped = int(df["stuck"].sum())
        df = df.loc[~df["stuck"]].reset_index(drop=True)

    reset_crash_rows_dropped = 0
    if drop_reset_crashes:
        reset_crash_rows_dropped = int(df["reset_crash"].sum())
        df = df.loc[~df["reset_crash"]].reset_index(drop=True)

    df["action"] = df.apply(lambda row: json.dumps([row["throttle"], row["steer"]]), axis=1)
    df["obs"] = df["obs_list"].apply(json.dumps)
    df["input_source"] = input_source
    df = df.drop(columns=["action_list", "obs_list", "throttle", "steer", "stuck", "reset_crash"])
    df["throttle"] = df["action"].apply(lambda a: json.loads(a)[0])
    df["steer"] = df["action"].apply(lambda a: json.loads(a)[1])

    stats = FileCleanStats(
        name=path.name,
        agent_id=agent_id,
        input_source=input_source,
        rows_before=rows_before,
        rows_after_dedup=rows_after_dedup,
        duplicates_dropped=dropped,
        conflict_groups=conflict_groups,
        conflict_rows_resolved=conflict_rows,
        stuck_segments=len(segments),
        stuck_rows=stuck_rows_total,
        stuck_rows_dropped=stuck_rows_dropped,
        manual_excluded_rows=manual_excluded_rows,
        reset_crash_segments=len(reset_segments),
        reset_crash_rows=reset_crash_rows_total,
        reset_crash_rows_dropped=reset_crash_rows_dropped,
    )
    return df, stats, segments, reset_segments


def assign_global_sample_weights(frames: list[pd.DataFrame]) -> list[pd.DataFrame]:
    by_agent: dict[int, list[int]] = {}
    for index, frame in enumerate(frames):
        agent_id = int(frame["agent_id"].iloc[0]) if len(frame) else -1
        by_agent.setdefault(agent_id, []).append(index)

    for agent_id, indices in by_agent.items():
        sizes = [len(frames[i]) for i in indices]
        throttle = np.concatenate([frames[i]["throttle"].to_numpy() for i in indices])
        steer = np.concatenate([frames[i]["steer"].to_numpy() for i in indices])
        weights = compute_sample_weights(throttle, steer)
        offset = 0
        for i, size in zip(indices, sizes):
            frames[i] = frames[i].copy()
            frames[i]["sample_weight"] = weights[offset : offset + size]
            offset += size
    return frames


def render_report(report: CleanReport) -> str:
    lines: list[str] = []
    lines.append("# Nettoyage du dataset Robocar")
    lines.append("")
    lines.append(
        f"`{len(report.files)}` fichiers traites, `{report.rows_before_total:,}` lignes avant, "
        f"`{report.rows_after_total:,}` lignes apres (suppression de doublons consecutifs uniquement; "
        f"les conflits de label sont resolus en place, pas supprimes).".replace(",", " ")
    )
    lines.append("")
    lines.append(
        "| Fichier | Agent | Source | Avant | Exclu manuellement | Apres dedup | Doublons supprimes | "
        "Groupes en conflit | Lignes lissees | Segments bloques | Lignes bloquees |"
    )
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for stats in sorted(report.files, key=lambda s: s.rows_before, reverse=True):
        lines.append(
            f"| `{stats.name}` | `{stats.agent_id}` | `{stats.input_source}` | `{stats.rows_before}` | "
            f"`{stats.manual_excluded_rows}` | `{stats.rows_after_dedup}` | `{stats.duplicates_dropped}` | "
            f"`{stats.conflict_groups}` | `{stats.conflict_rows_resolved}` | `{stats.stuck_segments}` | "
            f"`{stats.stuck_rows}` |"
        )
    lines.append("")
    by_source: dict[str, int] = {}
    for stats in report.files:
        by_source[stats.input_source] = by_source.get(stats.input_source, 0) + stats.rows_after_dedup
    lines.append("## Repartition par source d'entree")
    lines.append("")
    for source, rows in sorted(by_source.items()):
        lines.append(f"- `{source}`: `{rows:,}` lignes.".replace(",", " "))
    lines.append("")
    lines.append("## Segments suspects (potentiel blocage / crash)")
    lines.append("")
    if report.stuck_segments:
        lines.append(
            f"`{len(report.stuck_segments)}` segments detectes ou l'observation est restee quasi figee "
            f"pendant au moins `{STUCK_MIN_SECONDS:.1f}s` alors que le throttle restait au-dessus de "
            f"`{STUCK_MIN_THROTTLE}`. A valider manuellement ; relancer avec `--drop-stuck-segments` "
            "pour les retirer des CSV nettoyes une fois confirmes."
        )
        lines.append("")
        lines.append("| Fichier | Debut (timestamp) | Duree | Lignes | Throttle moyen | |steer| moyen |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for segment in sorted(report.stuck_segments, key=lambda s: s.duration, reverse=True):
            lines.append(
                f"| `{segment.file_name}` | `{segment.start_ts:.3f}` | `{segment.duration:.2f}s` | "
                f"`{segment.rows}` | `{segment.mean_throttle:.2f}` | `{segment.mean_abs_steer:.2f}` |"
            )
    else:
        lines.append("Aucun segment bloque detecte avec les seuils actuels.")
    lines.append("")
    lines.append("## Decisions ratees avant un retour au depart (agent 0 uniquement)")
    lines.append("")
    if report.reset_crash_segments:
        lines.append(
            f"`{len(report.reset_crash_segments)}` segments detectes : un rayon proche "
            f"(<= `{RESET_CLOSE_THRESHOLD}`) suivi, dans les `{RESET_LOOKBACK_SECONDS:.1f}s`, d'un saut "
            "brutal d'observation (teleport/reset). Les frames entre le contact et le reset sont listees ; "
            "relancer avec `--drop-reset-crashes` pour les retirer des CSV nettoyes une fois confirmees."
        )
        lines.append("")
        lines.append("| Fichier | Debut (timestamp) | Duree | Lignes | Rayon min |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for segment in sorted(report.reset_crash_segments, key=lambda s: s.min_ray_distance):
            lines.append(
                f"| `{segment.file_name}` | `{segment.start_ts:.3f}` | `{segment.duration:.2f}s` | "
                f"`{segment.rows}` | `{segment.min_ray_distance:.1f}` |"
            )
    else:
        lines.append("Aucun segment detecte avec les seuils actuels (ou aucune session d'agent 0 dans le lot).")
    lines.append("")
    lines.append("## Limites connues")
    lines.append("")
    lines.append(
        "- La resolution de conflit remplace les labels ambigus par leur moyenne par observation exacte, "
        "seulement quand l'ecart entre actions reste faible (bruit). Au-dela du seuil, le desaccord est "
        "considere comme un vrai choix divergent et n'est pas touche."
    )
    lines.append(
        "- Le `sample_weight` rebalance les bins (steer x throttle) par agent ; il ne corrige pas "
        "l'heterogeneite de capteurs entre agents (deja geree par l'entrainement separe par agent)."
    )
    lines.append(
        "- Le detecteur de blocage est une heuristique sur l'observation seule (raycasts quasi figes + "
        "throttle actif) : il rate les sorties de piste a vitesse normale et les chocs qui ne figent pas "
        "les raycasts, et peut occasionnellement flaguer un arret volontaire prolonge avec relance."
    )
    lines.append(
        "- Le detecteur de retour-au-depart n'est actif que pour agent 0 (capteur 10 rayons, fov 180). "
        "Sur agent 1 (36 rayons, fov etroit), une valeur sentinelle clignotante en bordure de champ de "
        "vision produit des faux positifs en masse avec le meme critere ; aucun seuil fiable trouve a ce jour."
    )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nettoyage des CSV Robocar avant entrainement")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Dossier des CSV bruts")
    parser.add_argument("--out-dir", type=Path, default=Path("data_clean"), help="Dossier des CSV nettoyes")
    parser.add_argument("--report", type=Path, default=Path("reports/data_clean.md"), help="Rapport Markdown")
    parser.add_argument(
        "--drop-stuck-segments",
        action="store_true",
        help="Retirer des CSV nettoyes les segments detectes comme bloques/crash (par defaut: lister seulement).",
    )
    parser.add_argument(
        "--drop-reset-crashes",
        action="store_true",
        help="Retirer des CSV nettoyes les frames detectees avant un retour au depart (agent 0 uniquement).",
    )
    parser.add_argument(
        "--exclusions",
        type=Path,
        default=Path("data_clean_exclusions.json"),
        help="JSON de plages [debut, fin] par fichier a exclure (produit par scripts/review_session.py).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(args.data_dir.glob("*.csv"))
    if not files:
        raise SystemExit(f"Aucun CSV trouve dans {args.data_dir}")

    manual_exclusions = load_manual_exclusions(args.exclusions)
    if manual_exclusions:
        print(f"Exclusions manuelles chargees depuis {args.exclusions} ({len(manual_exclusions)} fichier(s)).")

    cleaned_frames: list[pd.DataFrame] = []
    report = CleanReport()

    for path in files:
        df, stats, segments, reset_segments = clean_file(
            path,
            drop_stuck_segments=args.drop_stuck_segments,
            drop_reset_crashes=args.drop_reset_crashes,
            manual_exclusions=manual_exclusions,
        )
        cleaned_frames.append(df)
        report.files.append(stats)
        report.stuck_segments.extend(segments)
        report.reset_crash_segments.extend(reset_segments)
        report.rows_before_total += stats.rows_before
        report.rows_after_total += stats.rows_after_dedup

    cleaned_frames = assign_global_sample_weights(cleaned_frames)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    base_columns = list(REQUIRED_COLUMNS) + ["input_source", "sample_weight"]
    for path, frame in zip(files, cleaned_frames):
        # Colonnes en plus de REQUIRED_COLUMNS (ex: speed/target_speed du
        # materiel reel, cf ML_LIAM/real_car_client.py) : passees telles
        # quelles plutot que silencieusement supprimees.
        extra_columns = [c for c in frame.columns if c not in base_columns]
        frame[base_columns + extra_columns].to_csv(args.out_dir / path.name, index=False)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(report), encoding="utf-8")

    print(f"Fichiers nettoyes ecrits dans {args.out_dir}")
    print(f"Rapport: {args.report}")
    print(f"Lignes avant: {report.rows_before_total} | apres dedup: {report.rows_after_total}")


if __name__ == "__main__":
    main()
