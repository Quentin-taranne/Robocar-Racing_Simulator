# EDA du dataset Robocar

Rapport genere le 2026-04-19 18:57:40 a partir de `11` fichiers CSV dans `data/`.

## Resume executif

- Volume: `11` fichiers, `127 949` lignes, `2` agents.
- Schema: `obs` a longueur fixe `50`, `action` a longueur fixe `2`, aucune erreur de parsing detectee.
- Diversite effective: `57 989` observations uniques, soit `45.3%` des lignes.
- Redondance: `69 960` lignes dupliquent une observation deja vue (54.7%).
- Bruit de labels: `32 050` lignes portent des actions differentes pour une observation identique (25.0%).
- Verdict: dataset exploitable pour un premier modele supervise, mais avec desequilibres d'actions, forte autocorrelation et heterogeneite entre agents.

## Big, Clean, Diverse

| Axe | Verdict | Evidence |
| --- | --- | --- |
| Big | `OK pour prototype` | 127 949 lignes sur 11 sessions, environ 42.7 min de roulage. |
| Clean | `Structure propre, signaux pauvres` | 0 erreur de parsing, 0 timestamp non monotone, `done=1` sur 0 lignes. |
| Diverse | `A renforcer` | 45.3% d'observations uniques, steer=0 sur 88.9%, throttle=1 sur 85.6%. |

## Controles de schema

| Controle | Resultat |
| --- | --- |
| Colonnes attendues | `timestamp, episode, step, agent_id, reward, done, action, obs` |
| Fichiers avec colonnes manquantes | `0` |
| Erreurs de parsing JSON / types | `0` |
| Longueurs `obs` | `{50: 127949}` |
| Longueurs `action` | `{2: 127949}` |
| `done=1` | `0` lignes |
| Valeurs distinctes de `reward` | `[0.0]` |
| Valeurs distinctes de `episode` | `[0]` |
| Duree totale approx. | `2559.607s` (42.7 min) |

## Repartition par fichier

| Fichier | Lignes | Agents | Slots utiles | Debut | Fin | Duree approx. |
| --- | ---: | --- | --- | --- | --- | ---: |
| `drive_0_20260408-155743.csv` | `44 070` | `0` | `10:44070` | `2026-04-08 15:57:43` | `2026-04-08 16:12:24` | `881.474s` |
| `drive_0_20260408-094128.csv` | `14 488` | `0` | `10:14488` | `2026-04-08 09:41:28` | `2026-04-08 09:46:18` | `289.792s` |
| `drive_0_20260404-154915.csv` | `13 472` | `0` | `10:13472` | `2026-04-04 15:49:15` | `2026-04-04 15:53:45` | `269.519s` |
| `drive_1_20260405-151159.csv` | `12 392` | `1` | `36:12392` | `2026-04-05 15:11:59` | `2026-04-05 15:16:07` | `247.927s` |
| `drive_0_20260408-093737.csv` | `11 171` | `0` | `10:11171` | `2026-04-08 09:37:37` | `2026-04-08 09:41:20` | `223.467s` |
| `drive_1_20260406-133437.csv` | `10 884` | `1` | `36:10884` | `2026-04-06 13:34:37` | `2026-04-06 13:38:15` | `217.747s` |
| `drive_1_20260405-141438.csv` | `9 943` | `1` | `36:9943` | `2026-04-05 14:14:38` | `2026-04-05 14:17:57` | `198.892s` |
| `drive_1_20260405-143110.csv` | `7 033` | `1` | `36:7033` | `2026-04-05 14:31:10` | `2026-04-05 14:33:31` | `140.705s` |
| `drive_0_20260405-141302.csv` | `3 404` | `0` | `10:3404` | `2026-04-05 14:13:02` | `2026-04-05 14:14:10` | `68.170s` |
| `drive_1_20260405-164009.csv` | `768` | `1` | `36:768` | `2026-04-05 16:40:09` | `2026-04-05 16:40:24` | `15.404s` |
| `drive_0_20260405-151139.csv` | `324` | `0` | `10:324` | `2026-04-05 15:11:39` | `2026-04-05 15:11:46` | `6.510s` |

## Repartition par agent

| Agent | Lignes | Part | Slots utiles | Observations uniques | Lignes dupliquees |
| --- | ---: | ---: | --- | ---: | ---: |
| `0` | `86 929` | `67.9%` | `10:86929` | `39 329` | `47 600` |
| `1` | `41 020` | `32.1%` | `36:41020` | `18 660` | `22 360` |

## Distribution des actions

- `throttle=1.0`: `109 491` lignes (85.6%).
- `throttle=0.0`: `18 458` lignes (14.4%).
- `steer=0.0`: `113 778` lignes (88.9%).
- `steer=-1.0`: `10 723` lignes (8.4%).
- `steer=1.0`: `3 448` lignes (2.7%).

| Action `(throttle, steer)` | Lignes | Part |
| --- | ---: | ---: |
| `(1.0, 0.0)` | `96 960` | `75.8%` |
| `(0.0, 0.0)` | `16 818` | `13.1%` |
| `(1.0, -1.0)` | `9 618` | `7.5%` |
| `(1.0, 1.0)` | `2 913` | `2.3%` |
| `(0.0, -1.0)` | `1 105` | `0.9%` |
| `(0.0, 1.0)` | `535` | `0.4%` |

## Temporalite et redondance

- Pas de temps global: moyenne `0.020s`, mediane `0.020s`, p90 `0.042s`, max `0.190s`.
- Runs d'actions identiques: moyenne `20.08`, mediane `7.00`, p90 `42.00`, max `1 298` pas.
- Observations consecutives identiques: `61 176` transitions (47.8%).
- Nombre moyen de dimensions qui changent entre deux pas: `3.42`.
- Violations de monotonie: `timestamp` `0`, `step` non incrementaux `0`, resets `0`.

## Constats qualite

- Les agents n'utilisent pas le meme espace de capteurs: 10 valeurs utiles pour l'agent 0, 36 pour l'agent 1, avec padding a -1.
- Le dataset est fortement redondant: 54.7% des lignes repliquent une observation deja vue.
- La serie temporelle est tres autocorrellee: une grande part des pas consecutifs ont exactement la meme observation.
- La direction est tres desequilibree: steer=0 represente 88.9% des labels.
- L'acceleration est tres desequilibree: throttle=1 represente 85.6% des labels.
- Le bruit de labels est notable: 25.0% des lignes appartiennent a des observations associees a plusieurs actions.
- `reward` est constant a 0 sur tout le corpus et n'apporte aucun signal de qualite ou d'apprentissage.
- `done` n'apparait jamais a 1; les fins d'episode ne sont pas observees dans les fichiers fournis.
- `episode` est constant a 0; le corpus ne contient pas de segmentation exploitable par episode.
- Le volume brut est suffisant pour demarrer un modele supervise simple, mais la diversite effective est nettement plus faible que le nombre de lignes.

## Recommandations

- Entrainer separement par agent, ou ajouter explicitement `agent_id`, `nbRay` et `fov` comme features pour ne pas melanger deux distributions de capteurs.
- Reequilibrer la collecte avec davantage de virages a droite, a gauche, sorties de ligne, reprises de controle et zones a faible throttle.
- Sous-echantillonner les segments quasi statiques ou dedupliquer les observations consecutives identiques avant apprentissage.
- Utiliser une validation par fichier ou par session plutot qu'un `random_split`, sinon les doublons temporels rendent la validation trop optimiste.
- Verifier la logique de collecte des colonnes `done`, `reward` et `episode` si elles sont censees servir a autre chose qu'un simple logging.
- Confirmer cote simulateur pourquoi aucune transition terminale n'est capturee dans les CSV actuels.

## Notes

- Ratio de redondance observe: `0.547`. Ratio de conflits observation/action: `0.250`. Ratio d'observations consecutives identiques: `0.478`.
- Les CSV semblent propres au niveau structurel, mais ils ne reflettent pas encore des episodes annotables ni une couverture de conduite tres variee.
