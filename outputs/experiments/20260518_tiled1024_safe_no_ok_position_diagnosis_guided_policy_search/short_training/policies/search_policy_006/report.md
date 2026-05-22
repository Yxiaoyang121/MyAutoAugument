# Policy Search Short-Training: search_policy_006

- operations: `clahe(p=0.387, s=0.434), brightness(p=0.334, s=0.109), horizontal_flip(p=0.312, s=1.000), cutout(p=0.268, s=0.116), copy_paste(p=0.267, s=0.470)`
- train images / bboxes: `4602` / `6376`
- val images / bboxes: `677` / `905`
- Precision: `0.580`
- Recall: `0.689`
- mAP50: `0.672`
- mAP50-95: `0.453`
- balanced_score: `0.592150`
- recall_priority_score: `0.621200`
- map_priority_score: `0.567700`
- OOM: `false`
- training wall seconds: `2131.7`

## Per-Class Recall/AP50

| class id | class | Recall | AP50 |
| ---: | --- | ---: | ---: |
| 0 | OK2 | 1.000 | 0.995 |
| 1 | OK3 | 0.984 | 0.984 |
| 2 | 加强筋打伤 | 0.875 | 0.877 |
| 3 | 开裂 | 0.499 | 0.497 |
| 4 | 油污 | 0.444 | 0.255 |
| 5 | 浅划伤 | 0.538 | 0.508 |
| 6 | 漏背锡 | 0.630 | 0.519 |
| 7 | 碰伤 | 0.593 | 0.613 |
| 8 | 脏污 | 0.357 | 0.341 |
| 9 | 轮廓划伤 | 0.643 | 0.695 |
| 10 | 锡丝残留 | 0.833 | 0.806 |
| 11 | 锡尖 | 0.926 | 0.928 |
| 12 | 锡膏 | 0.630 | 0.714 |
