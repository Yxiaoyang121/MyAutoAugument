# Policy Search Short-Training: search_policy_030

- operations: `gamma(p=0.469, s=0.335), contrast(p=0.311, s=0.489), copy_paste(p=0.343, s=0.509), local_contrast(p=0.474, s=0.340), scale(p=0.559, s=0.090)`
- train images / bboxes: `4602` / `6421`
- val images / bboxes: `677` / `905`
- Precision: `0.655`
- Recall: `0.633`
- mAP50: `0.675`
- mAP50-95: `0.445`
- balanced_score: `0.591500`
- recall_priority_score: `0.609200`
- map_priority_score: `0.574800`
- OOM: `false`
- training wall seconds: `2143.9`

## Per-Class Recall/AP50

| class id | class | Recall | AP50 |
| ---: | --- | ---: | ---: |
| 0 | OK2 | 1.000 | 0.995 |
| 1 | OK3 | 0.996 | 0.987 |
| 2 | 加强筋打伤 | 0.875 | 0.876 |
| 3 | 开裂 | 0.381 | 0.373 |
| 4 | 油污 | 0.389 | 0.174 |
| 5 | 浅划伤 | 0.498 | 0.577 |
| 6 | 漏背锡 | 0.627 | 0.611 |
| 7 | 碰伤 | 0.491 | 0.624 |
| 8 | 脏污 | 0.310 | 0.405 |
| 9 | 轮廓划伤 | 0.548 | 0.737 |
| 10 | 锡丝残留 | 0.583 | 0.739 |
| 11 | 锡尖 | 0.963 | 0.968 |
| 12 | 锡膏 | 0.565 | 0.714 |
