# EDA du dataset Robocar

Rapport genere le 2026-06-27 16:19:22 a partir de `58` fichiers CSV dans `data/`.

## Resume executif

- Volume: `58` fichiers, `852 778` lignes, `2` agents.
- Schema: `obs` a longueur fixe `50`, `action` a longueur fixe `2`, aucune erreur de parsing detectee.
- Diversite effective: `441 348` observations uniques, soit `51.8%` des lignes.
- Redondance: `411 430` lignes dupliquent une observation deja vue (48.2%).
- Bruit de labels: `332 022` lignes portent des actions differentes pour une observation identique (38.9%).
- Verdict: dataset exploitable pour un premier modele supervise, mais avec desequilibres d'actions, forte autocorrelation et heterogeneite entre agents.

## Big, Clean, Diverse

| Axe | Verdict | Evidence |
| --- | --- | --- |
| Big | `OK pour prototype` | 852 778 lignes sur 58 sessions, environ 288.5 min de roulage. |
| Clean | `Structure propre, signaux pauvres` | 0 erreur de parsing, 0 timestamp non monotone, `done=1` sur 0 lignes. |
| Diverse | `A renforcer` | 51.8% d'observations uniques, steer=0 sur 81.2%, throttle=1 sur 49.2%. |

## Controles de schema

| Controle | Resultat |
| --- | --- |
| Colonnes attendues | `timestamp, episode, step, agent_id, reward, done, action, obs` |
| Fichiers avec colonnes manquantes | `0` |
| Erreurs de parsing JSON / types | `0` |
| Longueurs `obs` | `{50: 852778}` |
| Longueurs `action` | `{2: 852778}` |
| `done=1` | `0` lignes |
| Valeurs distinctes de `reward` | `[0.0]` |
| Valeurs distinctes de `episode` | `[0]` |
| Duree totale approx. | `17307.006s` (288.5 min) |

## Repartition par fichier

| Fichier | Lignes | Agents | Slots utiles | Debut | Fin | Duree approx. |
| --- | ---: | --- | --- | --- | --- | ---: |
| `drive_0_20260622-225044.csv` | `178 389` | `0` | `10:178389` | `2026-06-22 22:50:44` | `2026-06-22 23:50:12` | `3567.882s` |
| `drive_0_20260627-152555.csv` | `61 984` | `0` | `10:61984` | `2026-06-27 15:25:55` | `2026-06-27 15:46:38` | `1242.960s` |
| `drive_1_20260627-152555.csv` | `61 984` | `1` | `36:61984` | `2026-06-27 15:25:55` | `2026-06-27 15:46:38` | `1242.959s` |
| `drive_0_20260622-222858.csv` | `56 840` | `0` | `10:56840` | `2026-06-22 22:28:58` | `2026-06-22 22:47:55` | `1137.517s` |
| `drive_0_20260408-155743.csv` | `44 070` | `0` | `10:44070` | `2026-04-08 15:57:43` | `2026-04-08 16:12:24` | `881.474s` |
| `drive_0_20260627-160322.csv` | `41 591` | `0` | `10:41591` | `2026-06-27 16:03:22` | `2026-06-27 16:17:21` | `839.301s` |
| `drive_1_20260627-160322.csv` | `41 591` | `1` | `36:41591` | `2026-06-27 16:03:22` | `2026-06-27 16:17:21` | `839.300s` |
| `drive_0_20260627-143746.csv` | `25 365` | `0` | `10:25365` | `2026-06-27 14:37:46` | `2026-06-27 14:46:15` | `509.104s` |
| `drive_1_20260627-143746.csv` | `25 365` | `1` | `36:25365` | `2026-06-27 14:37:46` | `2026-06-27 14:46:15` | `509.103s` |
| `drive_0_20260627-145406.csv` | `19 889` | `0` | `10:19889` | `2026-06-27 14:54:06` | `2026-06-27 15:00:46` | `400.072s` |
| `drive_1_20260627-145406.csv` | `19 889` | `1` | `36:19889` | `2026-06-27 14:54:06` | `2026-06-27 15:00:46` | `400.071s` |
| `drive_0_20260627-144652.csv` | `19 547` | `0` | `10:19547` | `2026-06-27 14:46:52` | `2026-06-27 14:53:25` | `393.002s` |
| `drive_1_20260627-144652.csv` | `19 547` | `1` | `36:19547` | `2026-06-27 14:46:52` | `2026-06-27 14:53:25` | `393.002s` |
| `drive_0_20260622-222221.csv` | `17 023` | `0` | `10:17023` | `2026-06-22 22:22:21` | `2026-06-22 22:28:02` | `340.476s` |
| `drive_0_20260408-094128.csv` | `14 488` | `0` | `10:14488` | `2026-04-08 09:41:28` | `2026-04-08 09:46:18` | `289.792s` |
| `drive_0_20260627-140802.csv` | `14 459` | `0` | `10:14459` | `2026-06-27 14:08:02` | `2026-06-27 14:12:52` | `289.375s` |
| `drive_1_20260627-140802.csv` | `14 459` | `1` | `36:14459` | `2026-06-27 14:08:02` | `2026-06-27 14:12:52` | `289.370s` |
| `drive_0_20260623-092029.csv` | `14 296` | `0` | `10:14296` | `2026-06-23 09:20:29` | `2026-06-23 09:25:22` | `292.975s` |
| `drive_0_20260404-154915.csv` | `13 472` | `0` | `10:13472` | `2026-04-04 15:49:15` | `2026-04-04 15:53:45` | `269.519s` |
| `drive_0_20260627-141358.csv` | `12 698` | `0` | `10:12698` | `2026-06-27 14:13:58` | `2026-06-27 14:18:12` | `254.007s` |
| `drive_1_20260627-141358.csv` | `12 698` | `1` | `36:12698` | `2026-06-27 14:13:58` | `2026-06-27 14:18:12` | `254.007s` |
| `drive_1_20260405-151159.csv` | `12 392` | `1` | `36:12392` | `2026-04-05 15:11:59` | `2026-04-05 15:16:07` | `247.927s` |
| `drive_0_20260408-093737.csv` | `11 171` | `0` | `10:11171` | `2026-04-08 09:37:37` | `2026-04-08 09:41:20` | `223.467s` |
| `drive_1_20260406-133437.csv` | `10 884` | `1` | `36:10884` | `2026-04-06 13:34:37` | `2026-04-06 13:38:15` | `217.747s` |
| `drive_1_20260405-141438.csv` | `9 943` | `1` | `36:9943` | `2026-04-05 14:14:38` | `2026-04-05 14:17:57` | `198.892s` |
| `drive_0_20260623-092531.csv` | `9 914` | `0` | `10:9914` | `2026-06-23 09:25:31` | `2026-06-23 09:28:49` | `198.318s` |
| `drive_0_20260627-152154.csv` | `9 774` | `0` | `10:9774` | `2026-06-27 15:21:54` | `2026-06-27 15:25:14` | `200.126s` |
| `drive_1_20260627-152154.csv` | `9 774` | `1` | `36:9774` | `2026-06-27 15:21:54` | `2026-06-27 15:25:14` | `200.125s` |
| `drive_1_20260405-143110.csv` | `7 033` | `1` | `36:7033` | `2026-04-05 14:31:10` | `2026-04-05 14:33:31` | `140.705s` |
| `drive_1_20260419-190344.csv` | `6 362` | `1` | `36:6362` | `2026-04-19 19:03:44` | `2026-04-19 19:05:52` | `127.940s` |
| `drive_0_20260627-130330.csv` | `6 299` | `0` | `10:6299` | `2026-06-27 13:03:30` | `2026-06-27 13:05:36` | `126.079s` |
| `drive_1_20260627-130330.csv` | `6 299` | `1` | `36:6299` | `2026-06-27 13:03:30` | `2026-06-27 13:05:36` | `126.078s` |
| `drive_0_20260405-141302.csv` | `3 404` | `0` | `10:3404` | `2026-04-05 14:13:02` | `2026-04-05 14:14:10` | `68.170s` |
| `drive_0_20260627-143547.csv` | `2 509` | `0` | `10:2509` | `2026-06-27 14:35:47` | `2026-06-27 14:36:45` | `58.451s` |
| `drive_0_20260622-175413.csv` | `2 081` | `0` | `10:2081` | `2026-06-22 17:54:13` | `2026-06-22 17:54:42` | `29.109s` |
| `drive_0_20260627-160227.csv` | `1 546` | `0` | `10:1546` | `2026-06-27 16:02:27` | `2026-06-27 16:03:09` | `42.052s` |
| `drive_1_20260627-160227.csv` | `1 546` | `1` | `36:1546` | `2026-06-27 16:02:27` | `2026-06-27 16:03:09` | `42.052s` |
| `drive_0_20260622-220603.csv` | `1 307` | `0` | `10:1307` | `2026-06-22 22:06:03` | `2026-06-22 22:06:45` | `41.408s` |
| `drive_1_20260419-190309.csv` | `1 286` | `1` | `36:1286` | `2026-04-19 19:03:09` | `2026-04-19 19:03:35` | `26.433s` |
| `drive_0_20260622-221024.csv` | `1 244` | `0` | `10:1244` | `2026-06-22 22:10:24` | `2026-06-22 22:10:50` | `25.979s` |
| `drive_0_20260622-221123.csv` | `847` | `0` | `10:847` | `2026-06-22 22:11:23` | `2026-06-22 22:11:47` | `23.783s` |
| `drive_0_20260627-142250.csv` | `831` | `0` | `10:831` | `2026-06-27 14:22:50` | `2026-06-27 14:23:17` | `26.298s` |
| `drive_1_20260405-164009.csv` | `768` | `1` | `36:768` | `2026-04-05 16:40:09` | `2026-04-05 16:40:24` | `15.404s` |
| `drive_0_20260627-142514.csv` | `686` | `0` | `10:686` | `2026-06-27 14:25:14` | `2026-06-27 14:25:41` | `27.592s` |
| `drive_0_20260627-114706.csv` | `556` | `0` | `10:556` | `2026-06-27 11:47:06` | `2026-06-27 11:47:18` | `11.171s` |
| `drive_1_20260627-114706.csv` | `556` | `1` | `36:556` | `2026-06-27 11:47:06` | `2026-06-27 11:47:18` | `11.170s` |
| `drive_0_20260627-115315.csv` | `493` | `0` | `10:493` | `2026-06-27 11:53:15` | `2026-06-27 11:53:25` | `9.905s` |
| `drive_1_20260627-115315.csv` | `493` | `1` | `36:493` | `2026-06-27 11:53:15` | `2026-06-27 11:53:25` | `9.905s` |
| `drive_0_20260627-143057.csv` | `414` | `0` | `10:414` | `2026-06-27 14:30:57` | `2026-06-27 14:31:10` | `12.991s` |
| `drive_0_20260622-221942.csv` | `374` | `0` | `10:374` | `2026-06-22 22:19:42` | `2026-06-22 22:20:09` | `26.871s` |
| `drive_0_20260627-142442.csv` | `362` | `0` | `10:362` | `2026-06-27 14:24:42` | `2026-06-27 14:25:03` | `21.126s` |
| `drive_0_20260505-161226.csv` | `357` | `0` | `10:357` | `2026-05-05 16:12:26` | `2026-05-05 16:12:33` | `7.188s` |
| `drive_0_20260505-161142.csv` | `346` | `0` | `10:346` | `2026-05-05 16:11:42` | `2026-05-05 16:11:52` | `9.837s` |
| `drive_0_20260622-222120.csv` | `342` | `0` | `10:342` | `2026-06-22 22:21:20` | `2026-06-22 22:21:28` | `7.985s` |
| `drive_0_20260405-151139.csv` | `324` | `0` | `10:324` | `2026-04-05 15:11:39` | `2026-04-05 15:11:46` | `6.510s` |
| `drive_0_20260627-143200.csv` | `264` | `0` | `10:264` | `2026-06-27 14:32:00` | `2026-06-27 14:32:10` | `10.758s` |
| `drive_0_20260627-142911.csv` | `260` | `0` | `10:260` | `2026-06-27 14:29:11` | `2026-06-27 14:29:33` | `21.450s` |
| `drive_0_20260627-142610.csv` | `93` | `0` | `10:93` | `2026-06-27 14:26:10` | `2026-06-27 14:27:21` | `70.735s` |

## Repartition par agent

| Agent | Lignes | Part | Slots utiles | Observations uniques | Lignes dupliquees |
| --- | ---: | ---: | --- | ---: | ---: |
| `0` | `589 909` | `69.2%` | `10:589909` | `272 131` | `317 778` |
| `1` | `262 869` | `30.8%` | `36:262869` | `169 217` | `93 652` |

## Distribution des actions

- `throttle=1.0`: `419 198` lignes (49.2%).
- `throttle=0.0`: `112 396` lignes (13.2%).
- `steer=0.0`: `692 726` lignes (81.2%).
- `steer=-1.0`: `62 069` lignes (7.3%).
- `steer=1.0`: `23 247` lignes (2.7%).

| Action `(throttle, steer)` | Lignes | Part |
| --- | ---: | ---: |
| `(1.0, 0.0)` | `356 257` | `41.8%` |
| `(1.0, 0.0)` | `156 582` | `18.4%` |
| `(0.0, 0.0)` | `96 429` | `11.3%` |
| `(1.0, -1.0)` | `43 405` | `5.1%` |
| `(1.0, 1.0)` | `19 536` | `2.3%` |
| `(0.0, -1.0)` | `8 204` | `1.0%` |

## Temporalite et redondance

- Pas de temps global: moyenne `0.020s`, mediane `0.019s`, p90 `0.029s`, max `65.756s`.
- Runs d'actions identiques: moyenne `6.47`, mediane `1.00`, p90 `14.00`, max `2 611` pas.
- Observations consecutives identiques: `291 044` transitions (34.1%).
- Nombre moyen de dimensions qui changent entre deux pas: `4.27`.
- Violations de monotonie: `timestamp` `0`, `step` non incrementaux `0`, resets `0`.

## Constats qualite

- Les agents n'utilisent pas le meme espace de capteurs: 10 valeurs utiles pour l'agent 0, 36 pour l'agent 1, avec padding a -1.
- Le dataset est fortement redondant: 48.2% des lignes repliquent une observation deja vue.
- La direction est tres desequilibree: steer=0 represente 81.2% des labels.
- Le bruit de labels est notable: 38.9% des lignes appartiennent a des observations associees a plusieurs actions.
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

- Ratio de redondance observe: `0.482`. Ratio de conflits observation/action: `0.389`. Ratio d'observations consecutives identiques: `0.341`.
- Les CSV semblent propres au niveau structurel, mais ils ne reflettent pas encore des episodes annotables ni une couverture de conduite tres variee.
