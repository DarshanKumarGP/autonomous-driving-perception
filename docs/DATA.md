# Data notes (BDD100K subset)

## Source and licence
- Dataset: BDD100K (UC Berkeley DeepDrive), 100K-image object-detection task, box2d labels only.
- Obtained via the Hugging Face copy `dronefreak/BDD100K`, which says it was mirrored from the Kaggle upload `solesensei/solesensei_bdd100k`. It is an unofficial redistribution.
- Licence: BDD100K data licence (research / educational use, with notice). Credit: Yu et al., "BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning", CVPR 2020.
- Labels are the older 2018 box annotations, as stated on the Hugging Face dataset card.
- Full images are NOT stored in this repo. Only the plots and metadata are.

## Our subset (random sample, seed 42)
| Split | Images | Boxes | Source |
|---|---|---|---|
| train | 4,000 | 74,380 | official train, shard_000 |
| val (monitoring during training) | 800 | 15,115 | official train, shard_001 |
| test (held out, used only for final numbers) | 800 | 14,718 | official BDD100K val, shard_000 |

Labels were converted from per-image JSON (box2d corners, 1280x720) to YOLO text files (class, x_center, y_center, w, h, normalised). 41 boxes under 1 px wide or tall were dropped. The subset zip (324 MB) is stored in Google Drive, not in git.

## Class balance (train subset)
person 5,206 (7.0%), rider 234 (0.3%), car 41,341 (55.6%), truck 1,737 (2.3%), bus 642 (0.9%), train 12 (0.02%), motor 162 (0.2%), bike 422 (0.6%), traffic light 10,850 (14.6%), traffic sign 13,774 (18.5%).
See assets/class_balance_train.png.

## Conditions (all 5,600 images, from BDD100K attributes)
- Weather: clear 2,937; overcast 753; undefined 690; partly cloudy 432; snowy 409; rainy 371; foggy only a handful.
- Time of day: daytime 3,037; night 2,130; dawn/dusk 423; undefined 10.
See assets/conditions.png and docs/splits_metadata.csv.

## Known limits
- `train` has 12 training boxes and 1 test box, so its AP cannot be learned or measured reliably and will be reported as insufficient data. `rider` and `motor` are also rare, so their numbers will be noisy.
- Subset only (about 5% of the labelled images), so absolute scores are lower than full-dataset results.
- Fog is almost absent, so fog performance is not evaluated.
