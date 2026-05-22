# Policy Search Short-Training: search_policy_016

- operations: `copy_paste(p=0.256, s=0.429), clahe(p=0.617, s=0.345), local_contrast(p=0.572, s=0.402), horizontal_flip(p=0.444, s=1.000), contrast(p=0.580, s=0.107)`
- train images / bboxes: `4602` / `6380`
- val images / bboxes: `677` / `905`
- Precision: `0.650`
- Recall: `0.596`
- mAP50: `0.647`
- mAP50-95: `0.428`
- balanced_score: `0.569150`
- recall_priority_score: `0.583250`
- map_priority_score: `0.554900`
- OOM: `false`
- training wall seconds: `2330.3`

## Per-Class Recall/AP50

| class id | class | Recall | AP50 |
| ---: | --- | ---: | ---: |
| 0 | OK2 | 1.000 | 0.995 |
| 1 | OK3 | 0.971 | 0.972 |
| 2 | 加强筋打伤 | 0.875 | 0.919 |
| 3 | 开裂 | 0.000 | 0.284 |
| 4 | 油污 | 0.278 | 0.260 |
| 5 | 浅划伤 | 0.308 | 0.337 |
| 6 | 漏背锡 | 0.630 | 0.533 |
| 7 | 碰伤 | 0.478 | 0.655 |
| 8 | 脏污 | 0.452 | 0.468 |
| 9 | 轮廓划伤 | 0.643 | 0.718 |
| 10 | 锡丝残留 | 0.750 | 0.601 |
| 11 | 锡尖 | 0.734 | 0.929 |
| 12 | 锡膏 | 0.627 | 0.744 |
