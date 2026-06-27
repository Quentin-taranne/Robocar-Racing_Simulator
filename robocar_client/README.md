# Démarrage rapide Robocar (client Python)

## Prérequis
- Conda ou Miniforge pour recréer l'environnement `robocar39`
- Le simulateur `RacingSimulator.app` présent dans ce dossier

## Installation rapide
Environnement reproductible utilisé pour l'entraînement :

```bash
cd $(dirname "$0")/..
conda env create -f environment.yml
conda activate robocar39
python --version
```

La version Python attendue est `3.9.18`.

Alternative avec venv :

```bash
cd $(dirname "$0")/..
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r robocar_client/requirements.txt
```

ML-Agents 0.30 n'est pas compatible avec Python >=3.11.

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

Pour collecter Agent0 et Agent1 en même temps avec le même clavier :

```bash
ALL_BEHAVIORS=1 scripts/run_client.sh
```

Les CSV restent séparés par `agent_id` (`data/drive_0_*.csv`, `data/drive_1_*.csv`). Gardez un entraînement séparé par agent si les observations/capteurs ne sont pas identiques.

Raccourcis clavier :
- Flèches ou WASD pour gauche/droite/accélérer/freiner
- Ctrl+C ou fermer la petite fenêtre pygame pour quitter

Les traces sont écrites dans `data/drive_YYYYMMDD-HHMMSS.csv`.

## Entraîner un modèle supervisé
```bash
python robocar_client/train_model.py \
  data/drive_20260403-*.csv \
  --agent-id 0 \
  --epochs 20 \
  --batch-size 256 \
  --hidden-size 128
```
Le modèle est sauvegardé par défaut dans `models/steering_mlp.pt` et les métriques dans `models/steering_mlp.metrics.json`.

Le script utilise par défaut une validation par fichier CSV, ce qui évite une validation trop optimiste due aux doublons temporels. Les métriques exportées incluent notamment `MAE`, `RMSE`, `accuracy` sur actions discrétisées et comparaison contre des baselines simples.

## Faire l'EDA du dataset
```bash
python3 scripts/eda_dataset.py
```
Le rapport Markdown est généré dans `reports/data_eda.md` et le résumé brut dans `reports/data_eda.json`.

## Notes
- Le fichier `agents.json` définit `fov` et `nbRay` pour les agents. Ajustez-le avant de lancer le client.
- `--time-scale` permet d'accélérer le simulateur (ex: `--time-scale 2.0`).
- Le client envoie la même commande aux agents actifs ; suffisant pour démarrer rapidement.
- Le petit MLP suffit pour démarrer; avant d'augmenter sa taille, privilégier davantage de données propres et variées.
