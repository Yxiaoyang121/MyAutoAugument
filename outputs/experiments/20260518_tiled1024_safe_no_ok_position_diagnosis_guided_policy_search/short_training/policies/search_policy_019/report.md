# Policy Search Short-Training: search_policy_019

- operations: `contrast(p=0.416, s=0.231), clahe(p=0.427, s=0.459), gamma(p=0.552, s=0.118), scale(p=0.253, s=0.114), translate(p=0.280, s=0.204)`
- train images / bboxes: `4602` / `6362`
- val images / bboxes: `677` / `905`
- Precision: `0.667`
- Recall: `0.636`
- mAP50: `0.658`
- mAP50-95: `0.441`
- balanced_score: `0.589200`
- recall_priority_score: `0.607150`
- map_priority_score: `0.570800`
- OOM: `false`
- training wall seconds: `2298.6`

## Per-Class Recall/AP50

| class id | class | Recall | AP50 |
| ---: | --- | ---: | ---: |
| 0 | OK2 | 1.000 | 0.995 |
| 1 | OK3 | 0.949 | 0.992 |
| 2 | 加强筋打伤 | 0.875 | 0.904 |
| 3 | 开裂 | 0.472 | 0.497 |
| 4 | 油污 | 0.278 | 0.231 |
| 5 | 浅划伤 | 0.462 | 0.509 |
| 6 | 漏背锡 | 0.500 | 0.567 |
| 7 | 碰伤 | 0.513 | 0.611 |
| 8 | 脏污 | 0.357 | 0.385 |
| 9 | 轮廓划伤 | 0.583 | 0.704 |
| 10 | 锡丝残留 | 0.645 | 0.480 |
| 11 | 锡尖 | 0.955 | 0.950 |
| 12 | 锡膏 | 0.674 | 0.724 |
