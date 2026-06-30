# Nettoyage du dataset Robocar

`22` fichiers traites  `90 108` lignes avant  `74 103` lignes apres (suppression de doublons consecutifs uniquement; les conflits de label sont resolus en place  pas supprimes).

| Fichier | Agent | Source | Avant | Exclu manuellement | Apres dedup | Doublons supprimes | Groupes en conflit | Lignes lissees | Segments bloques | Lignes bloquees |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `drive_0_20260629-135402.csv` | `0` | `manette` | `7526` | `0` | `5821` | `1705` | `215` | `429` | `6` | `632` |
| `drive_1_20260629-135402.csv` | `1` | `manette` | `7526` | `0` | `6758` | `768` | `67` | `112` | `1` | `66` |
| `drive_0_20260629-135828.csv` | `0` | `manette` | `6649` | `0` | `5429` | `1220` | `193` | `440` | `4` | `378` |
| `drive_1_20260629-135828.csv` | `1` | `manette` | `6649` | `0` | `6106` | `543` | `55` | `105` | `0` | `0` |
| `drive_0_20260629-140144.csv` | `0` | `manette` | `5512` | `0` | `3949` | `1563` | `51` | `102` | `4` | `618` |
| `drive_1_20260629-140144.csv` | `1` | `manette` | `5512` | `0` | `4855` | `657` | `19` | `26` | `0` | `0` |
| `drive_0_20260629-181700.csv` | `0` | `manette` | `4481` | `0` | `3573` | `908` | `150` | `274` | `2` | `267` |
| `drive_1_20260629-181700.csv` | `1` | `manette` | `4481` | `0` | `4114` | `367` | `40` | `70` | `0` | `0` |
| `drive_0_20260629-181052.csv` | `0` | `manette` | `4163` | `0` | `3231` | `932` | `82` | `164` | `4` | `505` |
| `drive_1_20260629-181052.csv` | `1` | `manette` | `4163` | `0` | `3736` | `427` | `21` | `36` | `1` | `87` |
| `drive_0_20260629-180513.csv` | `0` | `manette` | `3499` | `0` | `2456` | `1043` | `76` | `152` | `6` | `615` |
| `drive_1_20260629-180513.csv` | `1` | `manette` | `3499` | `0` | `3082` | `417` | `24` | `47` | `0` | `0` |
| `drive_0_20260629-180710.csv` | `0` | `manette` | `2884` | `0` | `2056` | `828` | `58` | `102` | `2` | `280` |
| `drive_1_20260629-180710.csv` | `1` | `manette` | `2884` | `0` | `2578` | `306` | `27` | `36` | `0` | `0` |
| `drive_0_20260629-180818.csv` | `0` | `manette` | `2800` | `0` | `1980` | `820` | `69` | `135` | `2` | `340` |
| `drive_1_20260629-180818.csv` | `1` | `manette` | `2800` | `0` | `2449` | `351` | `16` | `18` | `0` | `0` |
| `drive_0_20260629-180945.csv` | `0` | `manette` | `2799` | `0` | `2027` | `772` | `82` | `195` | `3` | `319` |
| `drive_1_20260629-180945.csv` | `1` | `manette` | `2799` | `0` | `2440` | `359` | `31` | `52` | `1` | `56` |
| `drive_0_20260629-181420.csv` | `0` | `manette` | `2417` | `0` | `1756` | `661` | `50` | `98` | `0` | `0` |
| `drive_1_20260629-181420.csv` | `1` | `manette` | `2417` | `0` | `2088` | `329` | `15` | `22` | `0` | `0` |
| `drive_0_20260629-181604.csv` | `0` | `manette` | `2324` | `0` | `1629` | `695` | `51` | `96` | `3` | `338` |
| `drive_1_20260629-181604.csv` | `1` | `manette` | `2324` | `0` | `1990` | `334` | `12` | `22` | `0` | `0` |

## Repartition par source d'entree

- `manette`: `74 103` lignes.

## Segments suspects (potentiel blocage / crash)

`39` segments detectes ou l'observation est restee quasi figee pendant au moins `1.0s` alors que le throttle restait au-dessus de `0.1`. A valider manuellement ; relancer avec `--drop-stuck-segments` pour les retirer des CSV nettoyes une fois confirmes.

| Fichier | Debut (timestamp) | Duree | Lignes | Throttle moyen | |steer| moyen |
| --- | ---: | ---: | ---: | ---: | ---: |
| `drive_0_20260629-140144.csv` | `1782734508.149` | `6.19s` | `311` | `1.00` | `0.00` |
| `drive_0_20260629-135402.csv` | `1782734047.430` | `6.17s` | `310` | `1.00` | `0.00` |
| `drive_0_20260629-180818.csv` | `1782749302.658` | `5.58s` | `280` | `1.00` | `0.00` |
| `drive_0_20260629-180513.csv` | `1782749118.269` | `5.16s` | `259` | `1.00` | `0.00` |
| `drive_0_20260629-181052.csv` | `1782749519.382` | `4.86s` | `243` | `0.97` | `0.00` |
| `drive_0_20260629-180710.csv` | `1782749236.803` | `4.49s` | `226` | `1.00` | `0.00` |
| `drive_0_20260629-181604.csv` | `1782749769.946` | `4.19s` | `210` | `0.99` | `0.00` |
| `drive_0_20260629-181700.csv` | `1782749825.655` | `4.10s` | `206` | `1.00` | `0.00` |
| `drive_0_20260629-135828.csv` | `1782734314.068` | `3.99s` | `200` | `1.00` | `0.00` |
| `drive_0_20260629-140144.csv` | `1782734611.978` | `2.97s` | `150` | `1.00` | `0.00` |
| `drive_0_20260629-180945.csv` | `1782749391.653` | `2.62s` | `132` | `1.00` | `0.00` |
| `drive_0_20260629-180945.csv` | `1782749394.345` | `2.48s` | `125` | `1.00` | `0.00` |
| `drive_0_20260629-181052.csv` | `1782749461.931` | `1.97s` | `100` | `1.00` | `0.00` |
| `drive_0_20260629-140144.csv` | `1782734548.212` | `1.95s` | `99` | `1.00` | `0.00` |
| `drive_0_20260629-181052.csv` | `1782749483.196` | `1.87s` | `95` | `1.00` | `0.00` |
| `drive_0_20260629-180513.csv` | `1782749182.455` | `1.82s` | `92` | `1.00` | `0.00` |
| `drive_1_20260629-181052.csv` | `1782749482.798` | `1.72s` | `87` | `1.00` | `0.00` |
| `drive_0_20260629-180513.csv` | `1782749169.079` | `1.56s` | `79` | `1.00` | `0.00` |
| `drive_0_20260629-180513.csv` | `1782749139.572` | `1.48s` | `75` | `0.36` | `0.00` |
| `drive_0_20260629-181604.csv` | `1782749810.246` | `1.46s` | `75` | `1.00` | `0.01` |
| `drive_0_20260629-135402.csv` | `1782734192.273` | `1.45s` | `74` | `1.00` | `0.00` |
| `drive_0_20260629-135402.csv` | `1782734145.518` | `1.35s` | `69` | `1.00` | `0.00` |
| `drive_0_20260629-181052.csv` | `1782749463.947` | `1.32s` | `67` | `0.99` | `0.00` |
| `drive_0_20260629-135828.csv` | `1782734381.153` | `1.30s` | `65` | `1.00` | `0.00` |
| `drive_1_20260629-135402.csv` | `1782734069.587` | `1.28s` | `66` | `1.00` | `0.00` |
| `drive_0_20260629-135402.csv` | `1782734146.910` | `1.25s` | `64` | `1.00` | `0.00` |
| `drive_0_20260629-180945.csv` | `1782749407.618` | `1.21s` | `62` | `1.00` | `0.00` |
| `drive_0_20260629-181700.csv` | `1782749907.170` | `1.18s` | `61` | `1.00` | `0.01` |
| `drive_0_20260629-180818.csv` | `1782749320.688` | `1.17s` | `60` | `1.00` | `0.01` |
| `drive_0_20260629-135828.csv` | `1782734318.097` | `1.15s` | `59` | `1.00` | `0.00` |
| `drive_0_20260629-135402.csv` | `1782734095.457` | `1.15s` | `59` | `1.00` | `0.00` |
| `drive_0_20260629-180513.csv` | `1782749170.670` | `1.13s` | `57` | `1.00` | `0.00` |
| `drive_0_20260629-140144.csv` | `1782734536.669` | `1.11s` | `58` | `1.00` | `0.00` |
| `drive_1_20260629-180945.csv` | `1782749393.412` | `1.10s` | `56` | `1.00` | `0.00` |
| `drive_0_20260629-135402.csv` | `1782734191.144` | `1.09s` | `56` | `1.00` | `0.00` |
| `drive_0_20260629-180710.csv` | `1782749235.681` | `1.06s` | `54` | `1.00` | `0.00` |
| `drive_0_20260629-135828.csv` | `1782734404.219` | `1.06s` | `54` | `0.92` | `0.00` |
| `drive_0_20260629-180513.csv` | `1782749172.256` | `1.04s` | `53` | `1.00` | `0.00` |
| `drive_0_20260629-181604.csv` | `1782749797.394` | `1.03s` | `53` | `0.60` | `0.01` |

## Decisions ratees avant un retour au depart (agent 0 uniquement)

Aucun segment detecte avec les seuils actuels (ou aucune session d'agent 0 dans le lot).

## Limites connues

- La resolution de conflit remplace les labels ambigus par leur moyenne par observation exacte, seulement quand l'ecart entre actions reste faible (bruit). Au-dela du seuil, le desaccord est considere comme un vrai choix divergent et n'est pas touche.
- Le `sample_weight` rebalance les bins (steer x throttle) par agent ; il ne corrige pas l'heterogeneite de capteurs entre agents (deja geree par l'entrainement separe par agent).
- Le detecteur de blocage est une heuristique sur l'observation seule (raycasts quasi figes + throttle actif) : il rate les sorties de piste a vitesse normale et les chocs qui ne figent pas les raycasts, et peut occasionnellement flaguer un arret volontaire prolonge avec relance.
- Le detecteur de retour-au-depart n'est actif que pour agent 0 (capteur 10 rayons, fov 180). Sur agent 1 (36 rayons, fov etroit), une valeur sentinelle clignotante en bordure de champ de vision produit des faux positifs en masse avec le meme critere ; aucun seuil fiable trouve a ce jour.
