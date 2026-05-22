# Policy Search Short-Training: search_policy_011

- operations: `brightness(p=0.699, s=0.158), local_contrast(p=0.413, s=0.124), sharpen(p=0.483, s=0.181), clahe(p=0.491, s=0.399), copy_paste(p=0.595, s=0.274)`
- train images / bboxes: `4602` / `6417`
- val images / bboxes: `677` / `905`
- Precision: `0.613`
- Recall: `0.565`
- mAP50: `0.653`
- mAP50-95: `0.437`
- balanced_score: `0.558200`
- recall_priority_score: `0.568600`
- map_priority_score: `0.549800`
- OOM: `false`
- training wall seconds: `2239.9`

## Per-Class Recall/AP50

| class id | class | Recall | AP50 |
| ---: | --- | ---: | ---: |
| 0 | OK2 | 1.000 | 0.995 |
| 1 | OK3 | 0.980 | 0.990 |
| 2 | 加强筋打伤 | 0.875 | 0.885 |
| 3 | 开裂 | 0.000 | 0.398 |
| 4 | 油污 | 0.167 | 0.151 |
| 5 | 浅划伤 | 0.308 | 0.369 |
| 6 | 漏背锡 | 0.587 | 0.608 |
| 7 | 碰伤 | 0.472 | 0.616 |
| 8 | 脏污 | 0.357 | 0.477 |
| 9 | 轮廓划伤 | 0.500 | 0.629 |
| 10 | 锡丝残留 | 0.471 | 0.661 |
| 11 | 锡尖 | 0.926 | 0.966 |
| 12 | 锡膏 | 0.697 | 0.739 |
