# Policy Search Short-Training: search_policy_029

- operations: `translate(p=0.289, s=0.241), horizontal_flip(p=0.623, s=1.000), copy_paste(p=0.510, s=0.247), contrast(p=0.509, s=0.117), gamma(p=0.565, s=0.391)`
- train images / bboxes: `4602` / `6394`
- val images / bboxes: `677` / `905`
- Precision: `0.607`
- Recall: `0.684`
- mAP50: `0.679`
- mAP50-95: `0.461`
- balanced_score: `0.600450`
- recall_priority_score: `0.626600`
- map_priority_score: `0.577900`
- OOM: `false`
- training wall seconds: `2230.4`

## Per-Class Recall/AP50

| class id | class | Recall | AP50 |
| ---: | --- | ---: | ---: |
| 0 | OK2 | 1.000 | 0.995 |
| 1 | OK3 | 0.985 | 0.981 |
| 2 | 加强筋打伤 | 0.875 | 0.877 |
| 3 | 开裂 | 0.912 | 0.745 |
| 4 | 油污 | 0.222 | 0.167 |
| 5 | 浅划伤 | 0.329 | 0.471 |
| 6 | 漏背锡 | 0.543 | 0.559 |
| 7 | 碰伤 | 0.599 | 0.659 |
| 8 | 脏污 | 0.381 | 0.439 |
| 9 | 轮廓划伤 | 0.548 | 0.679 |
| 10 | 锡丝残留 | 0.917 | 0.625 |
| 11 | 锡尖 | 0.999 | 0.940 |
| 12 | 锡膏 | 0.583 | 0.691 |
