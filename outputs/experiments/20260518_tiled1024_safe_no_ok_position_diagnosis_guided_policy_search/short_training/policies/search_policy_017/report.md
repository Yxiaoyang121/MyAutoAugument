# Policy Search Short-Training: search_policy_017

- operations: `gaussian_noise(p=0.189, s=0.104), gamma(p=0.475, s=0.456), local_contrast(p=0.371, s=0.179), cutout(p=0.142, s=0.163), copy_paste(p=0.621, s=0.273)`
- train images / bboxes: `4602` / `6408`
- val images / bboxes: `677` / `905`
- Precision: `0.697`
- Recall: `0.691`
- mAP50: `0.720`
- mAP50-95: `0.468`
- balanced_score: `0.632550`
- recall_priority_score: `0.654550`
- map_priority_score: `0.611700`
- OOM: `false`
- training wall seconds: `2104.1`

## Per-Class Recall/AP50

| class id | class | Recall | AP50 |
| ---: | --- | ---: | ---: |
| 0 | OK2 | 1.000 | 0.995 |
| 1 | OK3 | 0.978 | 0.975 |
| 2 | 加强筋打伤 | 0.875 | 0.877 |
| 3 | 开裂 | 1.000 | 0.995 |
| 4 | 油污 | 0.389 | 0.237 |
| 5 | 浅划伤 | 0.462 | 0.566 |
| 6 | 漏背锡 | 0.550 | 0.563 |
| 7 | 碰伤 | 0.422 | 0.601 |
| 8 | 脏污 | 0.429 | 0.517 |
| 9 | 轮廓划伤 | 0.494 | 0.694 |
| 10 | 锡丝残留 | 0.833 | 0.625 |
| 11 | 锡尖 | 0.889 | 0.933 |
| 12 | 锡膏 | 0.662 | 0.776 |
