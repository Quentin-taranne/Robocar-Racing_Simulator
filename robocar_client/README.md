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

Pour collecter Agent0 et Agent1 en même temps avec la même source d'input :

```bash
ALL_BEHAVIORS=1 scripts/run_client.sh
```

Les CSV restent séparés par `agent_id` (`data/drive_0_*.csv`, `data/drive_1_*.csv`). Gardez un entraînement séparé par agent si les observations/capteurs ne sont pas identiques.

Pour piloter avec une manette :

```bash
INPUT_MODE=auto IDLE_THROTTLE=0.0 scripts/run_client.sh
```

Si les contrôles ne bougent pas la voiture, vérifiez d'abord ce que le client Python lit et envoie :

```bash
INPUT_MODE=auto INPUT_DEBUG=1 GAMEPAD_DEBUG=1 IDLE_THROTTLE=0.0 scripts/run_client.sh
```

Gardez la petite fenêtre pygame au premier plan avec `INPUT_MODE=auto` ou `INPUT_MODE=pygame`. `INPUT_MODE=global` nécessite l'autorisation Accessibilité macOS pour le terminal.

Si `INPUT_DEBUG=1` affiche toujours `steer=0.00 throttle=0.00 brake=0.00`, le souci est avant Unity : le mode d'input choisi ne reçoit aucun événement. Testez d'abord `INPUT_MODE=auto` en cliquant la petite fenêtre de contrôle. Pour `INPUT_MODE=global`, autorisez le terminal dans Accessibilité macOS puis redémarrez le terminal.

Mapping par défaut :
- stick gauche horizontal : direction
- gâchette droite : accélération
- gâchette gauche : frein
- bouton A / Croix : accélération si les axes de gâchettes ne sont pas disponibles
- bouton B / Rond : frein si les axes de gâchettes ne sont pas disponibles

Si le mapping SDL/macOS diffère :

```bash
INPUT_MODE=gamepad GAMEPAD_THROTTLE_AXIS=5 GAMEPAD_BRAKE_AXIS=4 scripts/run_client.sh
```

Si la manette ne répond pas, inspectez les axes/boutons détectés :

```bash
python scripts/debug_gamepad.py
```

Vous pouvez aussi afficher les valeurs brutes pendant la collecte :

```bash
INPUT_MODE=auto GAMEPAD_DEBUG=1 IDLE_THROTTLE=0.0 scripts/run_client.sh
```

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

Les observations sont normalisées pendant l'entraînement, et les statistiques sont sauvegardées dans `models/*.metrics.json`. Gardez le fichier `.metrics.json` à côté du modèle `.pt` pour l'inférence.

Ne testez pas un modèle si l'entraînement affiche un warning de baseline. En particulier, si `exact_action_accuracy` est pire que la baseline majoritaire, le modèle peut avoir une RMSE correcte tout en prédisant presque toujours la même action.

Avec une manette, les actions sont continues. Dans ce cas `exact_action_accuracy` n'est pas la bonne métrique; utilisez plutôt `action_within_0.10`, `action_within_0.25`, `MAE` et `RMSE`.

Pour Agent1, vérifiez que les CSV contiennent assez de virages gauche/droite et de récupérations. Si la majorité des lignes est `(throttle=1.0, steer=0.0)`, le modèle apprend surtout à aller tout droit.

## Faire l'EDA du dataset
```bash
python3 scripts/eda_dataset.py
```
Le rapport Markdown est généré dans `reports/data_eda.md` et le résumé brut dans `reports/data_eda.json`.

## Notes
- Le fichier `agents.json` définit `fov` et `nbRay` pour les agents. Ajustez-le avant de lancer le client.
- `--time-scale` permet d'accélérer le simulateur (ex: `--time-scale 2.0`).
- Le client peut envoyer la même commande aux agents actifs, mais gardez ces données seulement si chaque agent suit réellement une trajectoire valide.
- Le petit MLP suffit pour démarrer; avant d'augmenter sa taille, privilégier davantage de données propres et variées.
