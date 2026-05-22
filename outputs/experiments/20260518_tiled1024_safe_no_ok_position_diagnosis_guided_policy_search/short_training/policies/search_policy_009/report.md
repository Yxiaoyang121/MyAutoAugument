# Policy Search Short-Training: search_policy_009

- operations: `horizontal_flip(p=0.295, s=1.000), clahe(p=0.423, s=0.236), sharpen(p=0.245, s=0.117)`
- train images / bboxes: `4602` / `6361`
- val images / bboxes: `677` / `905`
- Precision: `0.646`
- Recall: `0.636`
- mAP50: `0.645`
- mAP50-95: `0.429`
- balanced_score: `0.578150`
- recall_priority_score: `0.598350`
- map_priority_score: `0.557900`
- OOM: `false`
- training wall seconds: `2135.0`

## Per-Class Recall/AP50

| class id | class | Recall | AP50 |
| ---: | --- | ---: | ---: |
| 0 | OK2 | 1.000 | 0.995 |
| 1 | OK3 | 0.967 | 0.988 |
| 2 | 加强筋打伤 | 0.875 | 0.877 |
| 3 | 开裂 | 0.391 | 0.284 |
| 4 | 油污 | 0.278 | 0.100 |
| 5 | 浅划伤 | 0.385 | 0.483 |
| 6 | 漏背锡 | 0.630 | 0.596 |
| 7 | 碰伤 | 0.498 | 0.615 |
| 8 | 脏污 | 0.405 | 0.464 |
| 9 | 轮廓划伤 | 0.524 | 0.671 |
| 10 | 锡丝残留 | 0.750 | 0.701 |
| 11 | 锡尖 | 1.000 | 0.950 |
| 12 | 锡膏 | 0.565 | 0.664 |
