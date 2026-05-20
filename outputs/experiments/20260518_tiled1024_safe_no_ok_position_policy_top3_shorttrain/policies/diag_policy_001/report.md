# Short-Training Report: diag_policy_001

- policy_id: `diag_policy_001`
- source_issue: `low_contrast_missed_defect`
- operations: `clahe(p=0.411, s=0.382), contrast(p=0.407, s=0.372), gamma(p=0.324, s=0.311), brightness(p=0.229, s=0.241)`
- contains_copy_paste: `false`
- train images / bboxes: `4602` / `6364`
- val images / bboxes: `677` / `905`
- Precision: `0.709`
- Recall: `0.672`
- mAP50: `0.714`
- mAP50-95: `0.479`
- small_object_recall: `0.6196943972835314`
- short_train_score: `0.609204`
- score formula: `0.30*mAP50 + 0.35*mAP50-95 + 0.20*Recall + 0.15*small_object_recall`
- OOM: `false`
- training wall seconds: `2170.5`

## Per-Class Recall/AP50

| class id | class | Recall | AP50 |
| ---: | --- | ---: | ---: |
| 0 | OK2 | 1.000 | 0.995 |
| 1 | OK3 | 0.982 | 0.994 |
| 2 | 加强筋打伤 | 0.875 | 0.878 |
| 3 | 开裂 | 1.000 | 0.995 |
| 4 | 油污 | 0.315 | 0.169 |
| 5 | 浅划伤 | 0.538 | 0.578 |
| 6 | 漏背锡 | 0.522 | 0.645 |
| 7 | 碰伤 | 0.425 | 0.614 |
| 8 | 脏污 | 0.452 | 0.493 |
| 9 | 轮廓划伤 | 0.476 | 0.643 |
| 10 | 锡丝残留 | 0.667 | 0.639 |
| 11 | 锡尖 | 0.926 | 0.926 |
| 12 | 锡膏 | 0.565 | 0.717 |
