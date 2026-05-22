# Policy Search Short-Training: search_policy_010

- operations: `gamma(p=0.594, s=0.298), copy_paste(p=0.602, s=0.261), sharpen(p=0.399, s=0.257), local_contrast(p=0.574, s=0.309)`
- train images / bboxes: `4602` / `6411`
- val images / bboxes: `677` / `905`
- Precision: `0.732`
- Recall: `0.572`
- mAP50: `0.697`
- mAP50-95: `0.477`
- balanced_score: `0.606750`
- recall_priority_score: `0.608250`
- map_priority_score: `0.603500`
- OOM: `false`
- training wall seconds: `2072.2`

## Per-Class Recall/AP50

| class id | class | Recall | AP50 |
| ---: | --- | ---: | ---: |
| 0 | OK2 | 1.000 | 0.995 |
| 1 | OK3 | 0.974 | 0.988 |
| 2 | 加强筋打伤 | 0.625 | 0.646 |
| 3 | 开裂 | 0.000 | 0.995 |
| 4 | 油污 | 0.425 | 0.246 |
| 5 | 浅划伤 | 0.462 | 0.530 |
| 6 | 漏背锡 | 0.491 | 0.627 |
| 7 | 碰伤 | 0.461 | 0.607 |
| 8 | 脏污 | 0.333 | 0.395 |
| 9 | 轮廓划伤 | 0.417 | 0.652 |
| 10 | 锡丝残留 | 0.750 | 0.688 |
| 11 | 锡尖 | 0.889 | 0.913 |
| 12 | 锡膏 | 0.613 | 0.781 |
