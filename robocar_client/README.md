# Démarrage rapide Robocar (client Python)

## Prérequis
- Python 3.10 (ML-Agents 0.30 n'est pas compatible avec Python ≥3.11)
- Le simulateur `RacingSimulator.app` présent dans ce dossier

## Installation rapide
```bash
cd $(dirname "$0")/..
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r robocar_client/requirements.txt
```

## Lancer le simulateur
Sur macOS :
```bash
open RacingSimulator.app
```
(ou lancez l'exécutable `RacingSimulator.app/Contents/MacOS/RacingSimulator` si vous préférez la ligne de commande).

## Piloter et collecter
Dans un second terminal activé avec la venv :
```bash
python robocar_client/client.py \
  --agents-config robocar_client/agents.json \
  --base-port 5004
```
Raccourcis clavier :
- Flèches ou WASD pour gauche/droite/accélérer/freiner
- Ctrl+C ou fermer la petite fenêtre pygame pour quitter

Les traces sont écrites dans `data/drive_YYYYMMDD-HHMMSS.csv`.

## Entraîner un modèle supervisé
```bash
python robocar_client/train_model.py data/drive_20260403-*.csv --epochs 20 --batch-size 256
```
Le modèle est sauvegardé par défaut dans `models/steering_mlp.pt`.

## Notes
- Le fichier `agents.json` définit `fov` et `nbRay` pour les agents. Ajustez-le avant de lancer le client.
- `--time-scale` permet d'accélérer le simulateur (ex: `--time-scale 2.0`).
- Le client envoie la même commande aux agents actifs ; suffisant pour démarrer rapidement.
