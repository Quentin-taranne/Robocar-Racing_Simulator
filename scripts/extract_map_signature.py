#!/usr/bin/env python3
"""
Extrait la signature de départ (obs au step 0) d'un CSV pour identifier une map
au moment de l'inférence.

Usage:
    python scripts/extract_map_signature.py \\
        --data data/drive_0_20260627-115315.csv \\
        --map-name "circuit_A" \\
        --output models/map_signatures.json

    # Lancer une fois par map. Les signatures s'accumulent dans le même JSON.
"""

import argparse
import ast
import csv
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Extrait la signature de départ d'une map depuis un CSV")
    p.add_argument("--data", required=True, type=Path, help="Fichier CSV enregistré sur cette map")
    p.add_argument("--map-name", required=True, help="Nom de la map (ex: 'circuit_A')")
    p.add_argument("--output", required=True, type=Path, help="Fichier JSON de signatures (créé ou mis à jour)")
    p.add_argument("--agent-id", type=int, default=None, help="Filtrer par agent_id (défaut: premier trouvé)")
    p.add_argument("--episode", type=int, default=0, help="Numéro d'épisode dont extraire le départ (défaut: 0)")
    args = p.parse_args()

    if not args.data.exists():
        raise SystemExit(f"Fichier non trouvé: {args.data}")

    obs_found = None
    with open(args.data, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if int(row["episode"]) != args.episode:
                continue
            if int(row["step"]) != 0:
                continue
            if args.agent_id is not None and int(row["agent_id"]) != args.agent_id:
                continue
            obs_found = ast.literal_eval(row["obs"])
            break

    if obs_found is None:
        raise SystemExit(
            f"Aucune ligne step=0 trouvée (episode={args.episode}, agent_id={args.agent_id}) dans {args.data}"
        )

    sigs: dict = {}
    if args.output.exists():
        try:
            sigs = json.loads(args.output.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    if args.map_name in sigs:
        print(f"[WARN] La map '{args.map_name}' existait déjà, signature remplacée.")

    sigs[args.map_name] = obs_found
    args.output.write_text(json.dumps(sigs, indent=2), encoding="utf-8")
    print(f"Signature '{args.map_name}' sauvegardée ({len(obs_found)} dims) -> {args.output}")
    print(f"Maps connues: {list(sigs.keys())}")


if __name__ == "__main__":
    main()
