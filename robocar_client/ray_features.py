"""Feature engineering partagee pour les observations raycast (train + inference).

Le capteur de l'agent 1 (fov etroit, 36 rayons) renvoie une valeur plafond
(~256) quand un rayon ne touche rien, avec une trainee de bruit juste au-dessus
(257-280) due a l'imprecision du retour "rien detecte". Une normalisation
brute traite ce plafond comme une distance comme une autre, alors que c'est un
signal categorique ("obstacle" vs "rien dans la portee du capteur") plaque sur
une distance continue. `expand_ray_hit_features` separe les deux : la distance
est plafonnee (plus de trainee de bruit au-dela du plafond), et un indicateur
binaire "rien detecte" est ajoute pour chaque rayon.

Ce module est importe par `train_model.py` (au chargement des sessions) et par
`inference_client.py` (sur l'observation live) pour garantir que la meme
transformation est appliquee des deux cotes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

NO_HIT_DISTANCE = 256.0  # plafond observe du capteur ; au-dela = "rien detecte"


def expand_ray_hit_features(
    obs: np.ndarray,
    n_rays: int,
    no_hit_distance: float = NO_HIT_DISTANCE,
) -> np.ndarray:
    """Remplace les `n_rays` premieres colonnes (distances brutes) par
    `[distances plafonnees..., indicateurs "rien detecte"...]`, en laissant le
    reste de l'observation (padding -1 inclus) inchange. Fonctionne sur un
    vecteur (1D, inference) ou un batch (2D, entrainement).
    """
    if n_rays <= 0:
        return np.asarray(obs, dtype=np.float32)
    obs = np.asarray(obs, dtype=np.float32)
    rays = obs[..., :n_rays]
    rest = obs[..., n_rays:]
    no_hit = (rays >= no_hit_distance).astype(np.float32)
    clipped = np.minimum(rays, no_hit_distance)
    return np.concatenate([clipped, no_hit, rest], axis=-1)


def compute_ray_velocity_batch(obs_sequence: np.ndarray, n_rays: int) -> np.ndarray:
    """Vitesse de rapprochement par rayon (delta entre pas consecutifs) sur une
    sequence temporelle COMPLETE et ORDONNEE d'une seule session (un fichier =
    une session continue, deja trie par timestamp). La toute premiere ligne
    n'a pas de "avant" : vitesse mise a 0 plutot que d'utiliser une session
    differente comme reference.

    Un MLP sans memoire ne voit qu'une frame a la fois : il ne peut pas savoir
    si un obstacle se rapproche vite ou lentement a partir d'une seule
    distance. Cette feature rend ce signal explicite au lieu de demander au
    modele de le deviner.
    """
    obs_sequence = np.asarray(obs_sequence, dtype=np.float32)
    if n_rays <= 0 or len(obs_sequence) == 0:
        return np.zeros((len(obs_sequence), max(n_rays, 0)), dtype=np.float32)
    rays = obs_sequence[:, :n_rays]
    velocity = np.zeros_like(rays)
    velocity[1:] = rays[1:] - rays[:-1]
    return velocity


def compute_ray_velocity_step(
    current_obs: np.ndarray,
    previous_obs: np.ndarray | None,
    n_rays: int,
) -> np.ndarray:
    """Equivalent pas-a-pas de `compute_ray_velocity_batch`, pour l'inference
    (un seul step a la fois, avec l'observation precedente fournie par
    l'appelant). `previous_obs=None` (premier step) -> vitesse nulle.
    """
    current_obs = np.asarray(current_obs, dtype=np.float32)
    if n_rays <= 0:
        return np.zeros((current_obs.shape[0], 0), dtype=np.float32)
    if previous_obs is None or previous_obs.shape != current_obs.shape:
        return np.zeros((current_obs.shape[0], n_rays), dtype=np.float32)
    return (current_obs[:, :n_rays] - previous_obs[:, :n_rays]).astype(np.float32)


def load_agent_n_rays(agents_config: Path, agent_id: int) -> int | None:
    """Lit `nbRay` pour `agent_id` dans agents.json (cf. robocar_client/agents.json)."""
    if not agents_config.exists():
        return None
    try:
        payload = json.loads(agents_config.read_text(encoding="utf-8"))
        agents = payload.get("agents", [])
        if 0 <= agent_id < len(agents):
            n_rays = agents[agent_id].get("nbRay")
            return int(n_rays) if n_rays else None
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    return None
