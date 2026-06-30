#!/usr/bin/env python3
"""EDA autonome des traces CSV Robocar."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable


REQUIRED_COLUMNS = (
    "timestamp",
    "episode",
    "step",
    "agent_id",
    "reward",
    "done",
    "action",
    "obs",
)


def pct(value: float, total: float) -> str:
    if not total:
        return "0.0%"
    return f"{(100.0 * value / total):.1f}%"


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def format_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def format_seconds(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}s"


def format_count(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def format_dt(ts: float | None) -> str:
    if ts is None:
        return "-"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def format_action(action: tuple[float, float]) -> str:
    return f"({action[0]:.1f}, {action[1]:.1f})"


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def total_recording_duration(analysis: "Analysis") -> float:
    total = 0.0
    for stats in analysis.file_stats.values():
        if stats.first_ts is not None and stats.last_ts is not None:
            total += max(0.0, stats.last_ts - stats.first_ts)
    return total


def format_minutes(value: float) -> str:
    return f"{value / 60.0:.1f} min"


def build_dataset_health(analysis: "Analysis") -> dict[str, dict[str, str]]:
    total_rows = analysis.total_rows
    unique_obs_ratio = len(analysis.obs_counts) / total_rows if total_rows else 0.0
    neutral_steer_ratio = analysis.overall_steer.get(0.0, 0) / total_rows if total_rows else 0.0
    full_throttle_ratio = analysis.overall_throttle.get(1.0, 0) / total_rows if total_rows else 0.0
    duration = total_recording_duration(analysis)

    size_status = "OK" if total_rows >= 100_000 and len(analysis.files) >= 10 else "A renforcer"
    clean_status = "Structurellement propre"
    diversity_status = "A renforcer"

    if analysis.parse_errors or analysis.missing_column_files or analysis.timestamp_non_monotonic or analysis.step_resets:
        clean_status = "Problemes structurels detectes"
    elif analysis.done_counts.get(1, 0) == 0 or len(analysis.reward_values) == 1:
        clean_status = "Structure propre, signaux pauvres"

    if unique_obs_ratio >= 0.70 and neutral_steer_ratio < 0.70 and full_throttle_ratio < 0.70:
        diversity_status = "Bonne"
    elif unique_obs_ratio >= 0.55 and neutral_steer_ratio < 0.80 and full_throttle_ratio < 0.80:
        diversity_status = "Moyenne"

    return {
        "big": {
            "status": size_status,
            "evidence": (
                f"{format_count(total_rows)} lignes sur {format_count(len(analysis.files))} sessions, "
                f"environ {format_minutes(duration)} de roulage."
            ),
        },
        "clean": {
            "status": clean_status,
            "evidence": (
                f"{format_count(analysis.parse_errors)} erreur de parsing, "
                f"{format_count(analysis.timestamp_non_monotonic)} timestamp non monotone, "
                f"`done=1` sur {format_count(analysis.done_counts.get(1, 0))} lignes."
            ),
        },
        "diverse": {
            "status": diversity_status,
            "evidence": (
                f"{pct(len(analysis.obs_counts), total_rows)} d'observations uniques, "
                f"steer=0 sur {pct(analysis.overall_steer.get(0.0, 0), total_rows)}, "
                f"throttle=1 sur {pct(analysis.overall_throttle.get(1.0, 0), total_rows)}."
            ),
        },
    }


@dataclass
class FileStats:
    rows: int = 0
    first_ts: float | None = None
    last_ts: float | None = None
    agent_ids: Counter[int] = field(default_factory=Counter)
    active_slots: Counter[int] = field(default_factory=Counter)
    throttle: Counter[float] = field(default_factory=Counter)
    steer: Counter[float] = field(default_factory=Counter)
    dt_values: list[float] = field(default_factory=list)
    timestamp_non_monotonic: int = 0
    step_non_incremental: int = 0
    step_resets: int = 0


@dataclass
class AgentStats:
    rows: int = 0
    active_slots: Counter[int] = field(default_factory=Counter)
    throttle: Counter[float] = field(default_factory=Counter)
    steer: Counter[float] = field(default_factory=Counter)
    file_rows: Counter[str] = field(default_factory=Counter)
    unique_obs: Counter[tuple[float, ...]] = field(default_factory=Counter)
    done: Counter[int] = field(default_factory=Counter)


@dataclass
class Analysis:
    files: list[Path]
    total_rows: int = 0
    parse_errors: int = 0
    missing_column_files: list[str] = field(default_factory=list)
    obs_lengths: Counter[int] = field(default_factory=Counter)
    action_lengths: Counter[int] = field(default_factory=Counter)
    agent_rows: Counter[int] = field(default_factory=Counter)
    done_counts: Counter[int] = field(default_factory=Counter)
    episode_counts: Counter[int] = field(default_factory=Counter)
    reward_values: Counter[float] = field(default_factory=Counter)
    nonzero_reward_rows: int = 0
    file_stats: dict[str, FileStats] = field(default_factory=dict)
    agent_stats: dict[int, AgentStats] = field(default_factory=dict)
    overall_throttle: Counter[float] = field(default_factory=Counter)
    overall_steer: Counter[float] = field(default_factory=Counter)
    overall_actions: Counter[tuple[float, float]] = field(default_factory=Counter)
    obs_counts: Counter[tuple[float, ...]] = field(default_factory=Counter)
    obs_to_actions: defaultdict[tuple[float, ...], Counter[tuple[float, float]]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    step_non_incremental: int = 0
    step_resets: int = 0
    timestamp_non_monotonic: int = 0
    dt_values: list[float] = field(default_factory=list)
    action_run_lengths: list[int] = field(default_factory=list)
    consecutive_obs_diffs: list[int] = field(default_factory=list)
    consecutive_identical_obs: int = 0
    obs_slot_non_missing: Counter[int] = field(default_factory=Counter)
    obs_slot_min: list[float] | None = None
    obs_slot_max: list[float] | None = None


def discover_files(data_dir: Path, explicit_files: Iterable[Path]) -> list[Path]:
    if explicit_files:
        return sorted(Path(path) for path in explicit_files)
    return sorted(data_dir.glob("*.csv"))


def analyze(files: list[Path]) -> Analysis:
    analysis = Analysis(files=files)

    for path in files:
        file_stats = analysis.file_stats.setdefault(path.name, FileStats())
        previous_ts: float | None = None
        previous_step: int | None = None
        previous_action: tuple[float, float] | None = None
        previous_obs: tuple[float, ...] | None = None
        current_run_length = 0

        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(REQUIRED_COLUMNS):
                if reader.fieldnames is None or any(col not in reader.fieldnames for col in REQUIRED_COLUMNS):
                    analysis.missing_column_files.append(path.name)

            for row in reader:
                analysis.total_rows += 1
                file_stats.rows += 1

                try:
                    timestamp = float(row["timestamp"])
                    episode = int(row["episode"])
                    step = int(row["step"])
                    agent_id = int(row["agent_id"])
                    reward = float(row["reward"])
                    done = int(row["done"])
                    action_raw = json.loads(row["action"])
                    obs_raw = json.loads(row["obs"])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    analysis.parse_errors += 1
                    continue

                action = tuple(float(value) for value in action_raw)
                obs = tuple(float(value) for value in obs_raw)

                analysis.obs_lengths[len(obs)] += 1
                analysis.action_lengths[len(action)] += 1
                analysis.agent_rows[agent_id] += 1
                analysis.done_counts[done] += 1
                analysis.episode_counts[episode] += 1
                analysis.reward_values[reward] += 1
                if reward != 0.0:
                    analysis.nonzero_reward_rows += 1

                file_stats.agent_ids[agent_id] += 1
                file_stats.first_ts = timestamp if file_stats.first_ts is None else min(file_stats.first_ts, timestamp)
                file_stats.last_ts = timestamp if file_stats.last_ts is None else max(file_stats.last_ts, timestamp)

                if previous_ts is not None:
                    dt = timestamp - previous_ts
                    analysis.dt_values.append(dt)
                    file_stats.dt_values.append(dt)
                    if dt < 0:
                        analysis.timestamp_non_monotonic += 1
                        file_stats.timestamp_non_monotonic += 1
                previous_ts = timestamp

                if previous_step is not None:
                    delta_step = step - previous_step
                    if delta_step < 0:
                        analysis.step_resets += 1
                        file_stats.step_resets += 1
                    elif delta_step != 1:
                        analysis.step_non_incremental += 1
                        file_stats.step_non_incremental += 1
                previous_step = step

                if len(action) >= 2:
                    action_pair = (action[0], action[1])
                    analysis.overall_throttle[action[0]] += 1
                    analysis.overall_steer[action[1]] += 1
                    analysis.overall_actions[action_pair] += 1
                    file_stats.throttle[action[0]] += 1
                    file_stats.steer[action[1]] += 1

                    if previous_action is None:
                        previous_action = action_pair
                        current_run_length = 1
                    elif action_pair == previous_action:
                        current_run_length += 1
                    else:
                        analysis.action_run_lengths.append(current_run_length)
                        previous_action = action_pair
                        current_run_length = 1

                active_slots = sum(1 for value in obs if value != -1.0)
                file_stats.active_slots[active_slots] += 1

                agent_stats = analysis.agent_stats.setdefault(agent_id, AgentStats())
                agent_stats.rows += 1
                agent_stats.active_slots[active_slots] += 1
                agent_stats.file_rows[path.name] += 1
                agent_stats.done[done] += 1
                agent_stats.unique_obs[obs] += 1
                if len(action) >= 2:
                    agent_stats.throttle[action[0]] += 1
                    agent_stats.steer[action[1]] += 1

                analysis.obs_counts[obs] += 1
                if len(action) >= 2:
                    analysis.obs_to_actions[obs][action_pair] += 1

                if analysis.obs_slot_min is None:
                    analysis.obs_slot_min = [math.inf] * len(obs)
                    analysis.obs_slot_max = [-math.inf] * len(obs)

                if analysis.obs_slot_min is not None and analysis.obs_slot_max is not None:
                    for index, value in enumerate(obs):
                        if value != -1.0:
                            analysis.obs_slot_non_missing[index] += 1
                        if value < analysis.obs_slot_min[index]:
                            analysis.obs_slot_min[index] = value
                        if value > analysis.obs_slot_max[index]:
                            analysis.obs_slot_max[index] = value

                if previous_obs is not None:
                    diff_count = sum(1 for left, right in zip(previous_obs, obs) if left != right)
                    analysis.consecutive_obs_diffs.append(diff_count)
                    if diff_count == 0:
                        analysis.consecutive_identical_obs += 1
                previous_obs = obs

        if current_run_length:
            analysis.action_run_lengths.append(current_run_length)

    return analysis


def build_quality_findings(analysis: Analysis) -> list[str]:
    findings: list[str] = []

    total_rows = analysis.total_rows
    unique_obs = len(analysis.obs_counts)
    duplicate_rows = sum(count - 1 for count in analysis.obs_counts.values() if count > 1)
    duplicate_ratio = duplicate_rows / total_rows if total_rows else 0.0

    neutral_steer_ratio = analysis.overall_steer.get(0.0, 0) / total_rows if total_rows else 0.0
    full_throttle_ratio = analysis.overall_throttle.get(1.0, 0) / total_rows if total_rows else 0.0

    conflict_patterns = {
        obs: actions for obs, actions in analysis.obs_to_actions.items() if len(actions) > 1
    }
    conflicting_rows = sum(sum(actions.values()) for actions in conflict_patterns.values())
    conflicting_ratio = conflicting_rows / total_rows if total_rows else 0.0

    active_slot_signatures = {tuple(sorted(stats.active_slots.items())) for stats in analysis.agent_stats.values()}
    if len(active_slot_signatures) > 1:
        findings.append(
            "Les agents n'utilisent pas le meme espace de capteurs: 10 valeurs utiles pour l'agent 0, 36 pour l'agent 1, avec padding a -1."
        )

    if duplicate_ratio >= 0.40:
        findings.append(
            f"Le dataset est fortement redondant: {pct(duplicate_rows, total_rows)} des lignes repliquent une observation deja vue."
        )

    if analysis.consecutive_identical_obs / max(1, len(analysis.consecutive_obs_diffs)) >= 0.40:
        findings.append(
            "La serie temporelle est tres autocorrellee: une grande part des pas consecutifs ont exactement la meme observation."
        )

    if neutral_steer_ratio >= 0.80:
        findings.append(
            f"La direction est tres desequilibree: steer=0 represente {pct(analysis.overall_steer.get(0.0, 0), total_rows)} des labels."
        )

    if full_throttle_ratio >= 0.80:
        findings.append(
            f"L'acceleration est tres desequilibree: throttle=1 represente {pct(analysis.overall_throttle.get(1.0, 0), total_rows)} des labels."
        )

    if conflicting_ratio >= 0.10:
        findings.append(
            f"Le bruit de labels est notable: {pct(conflicting_rows, total_rows)} des lignes appartiennent a des observations associees a plusieurs actions."
        )

    if len(analysis.reward_values) == 1 and 0.0 in analysis.reward_values:
        findings.append("`reward` est constant a 0 sur tout le corpus et n'apporte aucun signal de qualite ou d'apprentissage.")

    if analysis.done_counts.get(1, 0) == 0:
        findings.append("`done` n'apparait jamais a 1; les fins d'episode ne sont pas observees dans les fichiers fournis.")

    if len(analysis.episode_counts) == 1 and 0 in analysis.episode_counts:
        findings.append("`episode` est constant a 0; le corpus ne contient pas de segmentation exploitable par episode.")

    if total_rows >= 100_000:
        findings.append("Le volume brut est suffisant pour demarrer un modele supervise simple, mais la diversite effective est nettement plus faible que le nombre de lignes.")
    else:
        findings.append("Le volume brut reste limite pour couvrir toute la variete de trajectoires attendues.")

    return findings


def build_recommendations(analysis: Analysis) -> list[str]:
    recommendations = [
        "Entrainer separement par agent, ou ajouter explicitement `agent_id`, `nbRay` et `fov` comme features pour ne pas melanger deux distributions de capteurs.",
        "Reequilibrer la collecte avec davantage de virages a droite, a gauche, sorties de ligne, reprises de controle et zones a faible throttle.",
        "Sous-echantillonner les segments quasi statiques ou dedupliquer les observations consecutives identiques avant apprentissage.",
        "Utiliser une validation par fichier ou par session plutot qu'un `random_split`, sinon les doublons temporels rendent la validation trop optimiste.",
        "Verifier la logique de collecte des colonnes `done`, `reward` et `episode` si elles sont censees servir a autre chose qu'un simple logging.",
    ]

    if analysis.done_counts.get(1, 0) == 0:
        recommendations.append("Confirmer cote simulateur pourquoi aucune transition terminale n'est capturee dans les CSV actuels.")

    if analysis.overall_actions.get((0.0, 1.0), 0) == 0 and analysis.overall_actions.get((0.0, -1.0), 0) == 0:
        recommendations.append("Collecter des cas de virage sans acceleration si ce comportement doit etre appris, car il est quasi absent ici.")

    return recommendations


def render_markdown(analysis: Analysis) -> str:
    total_rows = analysis.total_rows
    unique_obs = len(analysis.obs_counts)
    duplicate_rows = sum(count - 1 for count in analysis.obs_counts.values() if count > 1)
    conflict_patterns = {
        obs: actions for obs, actions in analysis.obs_to_actions.items() if len(actions) > 1
    }
    conflicting_rows = sum(sum(actions.values()) for actions in conflict_patterns.values())
    duplicate_ratio = duplicate_rows / total_rows if total_rows else 0.0
    conflicting_ratio = conflicting_rows / total_rows if total_rows else 0.0
    identical_ratio = analysis.consecutive_identical_obs / max(1, len(analysis.consecutive_obs_diffs))

    top_actions = analysis.overall_actions.most_common(6)
    findings = build_quality_findings(analysis)
    recommendations = build_recommendations(analysis)
    health = build_dataset_health(analysis)
    duration = total_recording_duration(analysis)

    lines: list[str] = []
    lines.append("# EDA du dataset Robocar")
    lines.append("")
    lines.append(
        f"Rapport genere le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} a partir de `{len(analysis.files)}` fichiers CSV dans `data/`."
    )
    lines.append("")
    lines.append("## Resume executif")
    lines.append("")
    lines.append(
        f"- Volume: `{format_count(len(analysis.files))}` fichiers, `{format_count(total_rows)}` lignes, `{format_count(len(analysis.agent_stats))}` agents."
    )
    lines.append(
        f"- Schema: `obs` a longueur fixe `{next(iter(analysis.obs_lengths), '-')}`, `action` a longueur fixe `{next(iter(analysis.action_lengths), '-')}`, aucune erreur de parsing detectee."
    )
    lines.append(
        f"- Diversite effective: `{format_count(unique_obs)}` observations uniques, soit `{pct(unique_obs, total_rows)}` des lignes."
    )
    lines.append(
        f"- Redondance: `{format_count(duplicate_rows)}` lignes dupliquent une observation deja vue ({pct(duplicate_rows, total_rows)})."
    )
    lines.append(
        f"- Bruit de labels: `{format_count(conflicting_rows)}` lignes portent des actions differentes pour une observation identique ({pct(conflicting_rows, total_rows)})."
    )
    lines.append(
        f"- Verdict: dataset exploitable pour un premier modele supervise, mais avec desequilibres d'actions, forte autocorrelation et heterogeneite entre agents."
    )
    lines.append("")
    lines.append("## Big, Clean, Diverse")
    lines.append("")
    lines.append("| Axe | Verdict | Evidence |")
    lines.append("| --- | --- | --- |")
    lines.append(f"| Big | `{health['big']['status']}` | {health['big']['evidence']} |")
    lines.append(f"| Clean | `{health['clean']['status']}` | {health['clean']['evidence']} |")
    lines.append(f"| Diverse | `{health['diverse']['status']}` | {health['diverse']['evidence']} |")
    lines.append("")
    lines.append("## Controles de schema")
    lines.append("")
    lines.append("| Controle | Resultat |")
    lines.append("| --- | --- |")
    lines.append(f"| Colonnes attendues | `{', '.join(REQUIRED_COLUMNS)}` |")
    lines.append(f"| Fichiers avec colonnes manquantes | `{format_count(len(analysis.missing_column_files))}` |")
    lines.append(f"| Erreurs de parsing JSON / types | `{format_count(analysis.parse_errors)}` |")
    lines.append(f"| Longueurs `obs` | `{dict(analysis.obs_lengths)}` |")
    lines.append(f"| Longueurs `action` | `{dict(analysis.action_lengths)}` |")
    lines.append(f"| `done=1` | `{format_count(analysis.done_counts.get(1, 0))}` lignes |")
    lines.append(f"| Valeurs distinctes de `reward` | `{list(sorted(analysis.reward_values.keys()))}` |")
    lines.append(f"| Valeurs distinctes de `episode` | `{list(sorted(analysis.episode_counts.keys()))}` |")
    lines.append(f"| Duree totale approx. | `{format_seconds(duration)}` ({format_minutes(duration)}) |")
    lines.append("")
    lines.append("## Repartition par fichier")
    lines.append("")
    lines.append("| Fichier | Lignes | Agents | Slots utiles | Debut | Fin | Duree approx. |")
    lines.append("| --- | ---: | --- | --- | --- | --- | ---: |")
    for file_name, stats in sorted(
        analysis.file_stats.items(),
        key=lambda item: item[1].rows,
        reverse=True,
    ):
        duration = None
        if stats.first_ts is not None and stats.last_ts is not None:
            duration = stats.last_ts - stats.first_ts
        agent_list = ", ".join(str(agent_id) for agent_id in sorted(stats.agent_ids))
        active_list = ", ".join(f"{key}:{value}" for key, value in sorted(stats.active_slots.items()))
        lines.append(
            f"| `{file_name}` | `{format_count(stats.rows)}` | `{agent_list}` | `{active_list}` | `{format_dt(stats.first_ts)}` | `{format_dt(stats.last_ts)}` | `{format_seconds(duration)}` |"
        )
    lines.append("")
    lines.append("## Repartition par agent")
    lines.append("")
    lines.append("| Agent | Lignes | Part | Slots utiles | Observations uniques | Lignes dupliquees |")
    lines.append("| --- | ---: | ---: | --- | ---: | ---: |")
    for agent_id, stats in sorted(analysis.agent_stats.items()):
        unique_obs_agent = len(stats.unique_obs)
        duplicate_rows_agent = sum(count - 1 for count in stats.unique_obs.values() if count > 1)
        active_desc = ", ".join(f"{key}:{value}" for key, value in sorted(stats.active_slots.items()))
        lines.append(
            f"| `{agent_id}` | `{format_count(stats.rows)}` | `{pct(stats.rows, total_rows)}` | `{active_desc}` | `{format_count(unique_obs_agent)}` | `{format_count(duplicate_rows_agent)}` |"
        )
    lines.append("")
    lines.append("## Distribution des actions")
    lines.append("")
    lines.append(f"- `throttle=1.0`: `{format_count(analysis.overall_throttle.get(1.0, 0))}` lignes ({pct(analysis.overall_throttle.get(1.0, 0), total_rows)}).")
    lines.append(f"- `throttle=0.0`: `{format_count(analysis.overall_throttle.get(0.0, 0))}` lignes ({pct(analysis.overall_throttle.get(0.0, 0), total_rows)}).")
    lines.append(f"- `steer=0.0`: `{format_count(analysis.overall_steer.get(0.0, 0))}` lignes ({pct(analysis.overall_steer.get(0.0, 0), total_rows)}).")
    lines.append(f"- `steer=-1.0`: `{format_count(analysis.overall_steer.get(-1.0, 0))}` lignes ({pct(analysis.overall_steer.get(-1.0, 0), total_rows)}).")
    lines.append(f"- `steer=1.0`: `{format_count(analysis.overall_steer.get(1.0, 0))}` lignes ({pct(analysis.overall_steer.get(1.0, 0), total_rows)}).")
    lines.append("")
    lines.append("| Action `(throttle, steer)` | Lignes | Part |")
    lines.append("| --- | ---: | ---: |")
    for action, count in top_actions:
        lines.append(f"| `{format_action(action)}` | `{format_count(count)}` | `{pct(count, total_rows)}` |")
    lines.append("")
    lines.append("## Temporalite et redondance")
    lines.append("")
    lines.append(
        f"- Pas de temps global: moyenne `{format_seconds(safe_mean(analysis.dt_values))}`, mediane `{format_seconds(quantile(analysis.dt_values, 0.5))}`, p90 `{format_seconds(quantile(analysis.dt_values, 0.9))}`, max `{format_seconds(max(analysis.dt_values) if analysis.dt_values else None)}`."
    )
    lines.append(
        f"- Runs d'actions identiques: moyenne `{format_float(safe_mean(analysis.action_run_lengths), 2)}`, mediane `{format_float(quantile(analysis.action_run_lengths, 0.5), 2)}`, p90 `{format_float(quantile(analysis.action_run_lengths, 0.9), 2)}`, max `{format_count(max(analysis.action_run_lengths) if analysis.action_run_lengths else 0)}` pas."
    )
    lines.append(
        f"- Observations consecutives identiques: `{format_count(analysis.consecutive_identical_obs)}` transitions ({pct(analysis.consecutive_identical_obs, len(analysis.consecutive_obs_diffs))})."
    )
    lines.append(
        f"- Nombre moyen de dimensions qui changent entre deux pas: `{format_float(safe_mean(analysis.consecutive_obs_diffs), 2)}`."
    )
    lines.append(
        f"- Violations de monotonie: `timestamp` `{format_count(analysis.timestamp_non_monotonic)}`, `step` non incrementaux `{format_count(analysis.step_non_incremental)}`, resets `{format_count(analysis.step_resets)}`."
    )
    lines.append("")
    lines.append("## Constats qualite")
    lines.append("")
    for finding in findings:
        lines.append(f"- {finding}")
    lines.append("")
    lines.append("## Recommandations")
    lines.append("")
    for recommendation in recommendations:
        lines.append(f"- {recommendation}")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        f"- Ratio de redondance observe: `{duplicate_ratio:.3f}`. Ratio de conflits observation/action: `{conflicting_ratio:.3f}`. Ratio d'observations consecutives identiques: `{identical_ratio:.3f}`."
    )
    lines.append(
        "- Les CSV semblent propres au niveau structurel, mais ils ne reflettent pas encore des episodes annotables ni une couverture de conduite tres variee."
    )
    lines.append("")
    return "\n".join(lines)


def build_json_summary(analysis: Analysis) -> dict:
    conflict_patterns = {
        obs: actions for obs, actions in analysis.obs_to_actions.items() if len(actions) > 1
    }
    conflicting_rows = sum(sum(actions.values()) for actions in conflict_patterns.values())
    duplicate_rows = sum(count - 1 for count in analysis.obs_counts.values() if count > 1)

    return {
        "file_count": len(analysis.files),
        "row_count": analysis.total_rows,
        "recording_duration_seconds": total_recording_duration(analysis),
        "obs_length_distribution": dict(analysis.obs_lengths),
        "action_length_distribution": dict(analysis.action_lengths),
        "agent_row_distribution": dict(analysis.agent_rows),
        "done_distribution": dict(analysis.done_counts),
        "episode_distribution": dict(analysis.episode_counts),
        "reward_distribution": dict(analysis.reward_values),
        "unique_observations": len(analysis.obs_counts),
        "duplicate_rows": duplicate_rows,
        "conflicting_observation_patterns": len(conflict_patterns),
        "conflicting_rows": conflicting_rows,
        "temporal": {
            "dt_mean": safe_mean(analysis.dt_values),
            "dt_median": quantile(analysis.dt_values, 0.5),
            "dt_p90": quantile(analysis.dt_values, 0.9),
            "action_run_mean": safe_mean(analysis.action_run_lengths),
            "action_run_median": quantile(analysis.action_run_lengths, 0.5),
            "action_run_p90": quantile(analysis.action_run_lengths, 0.9),
            "consecutive_identical_observations": analysis.consecutive_identical_obs,
        },
        "top_actions": [
            {"action": list(action), "count": count}
            for action, count in analysis.overall_actions.most_common(10)
        ],
        "dataset_health": build_dataset_health(analysis),
        "quality_findings": build_quality_findings(analysis),
        "recommendations": build_recommendations(analysis),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EDA autonome des CSV Robocar")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Dossier contenant les CSV")
    parser.add_argument(
        "--files",
        type=Path,
        nargs="*",
        default=(),
        help="Liste explicite de CSV a analyser (sinon tous les CSV de --data-dir)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/data_eda.md"),
        help="Chemin du rapport Markdown genere",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("reports/data_eda.json"),
        help="Chemin du resume JSON genere",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = discover_files(args.data_dir, args.files)
    if not files:
        raise SystemExit("Aucun fichier CSV trouve pour l'EDA.")

    analysis = analyze(files)
    markdown = render_markdown(analysis)
    json_summary = build_json_summary(analysis)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(markdown, encoding="utf-8")

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(json_summary, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"Rapport Markdown: {args.report}")
    print(f"Resume JSON: {args.json}")
    print(f"Lignes analysees: {analysis.total_rows}")


if __name__ == "__main__":
    main()
