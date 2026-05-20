# Short-Training Report: diag_policy_002

- policy_id: `diag_policy_002`
- source_issue: `low_contrast_missed_defect`
- operations: `clahe(p=0.391, s=0.382), contrast(p=0.423, s=0.372), gamma(p=0.321, s=0.311), brightness(p=0.232, s=0.241)`
- contains_copy_paste: `false`
- train images / bboxes: `4602` / `6364`
- val images / bboxes: `677` / `905`
- Precision: `0.726`
- Recall: `0.613`
- mAP50: `0.710`
- mAP50-95: `0.460`
- small_object_recall: `0.6587436332767402`
- short_train_score: `0.595412`
- score formula: `0.30*mAP50 + 0.35*mAP50-95 + 0.20*Recall + 0.15*small_object_recall`
- OOM: `false`
- training wall seconds: `2084.1`

## Per-Class Recall/AP50

| class id | class | Recall | AP50 |
| ---: | --- | ---: | ---: |
| 0 | OK2 | 1.000 | 0.995 |
| 1 | OK3 | 0.967 | 0.992 |
| 2 | 加强筋打伤 | 0.875 | 0.971 |
| 3 | 开裂 | 0.500 | 0.695 |
| 4 | 油污 | 0.222 | 0.265 |
| 5 | 浅划伤 | 0.462 | 0.521 |
| 6 | 漏背锡 | 0.581 | 0.599 |
| 7 | 碰伤 | 0.520 | 0.653 |
| 8 | 脏污 | 0.333 | 0.421 |
| 9 | 轮廓划伤 | 0.499 | 0.735 |
| 10 | 锡丝残留 | 0.417 | 0.711 |
| 11 | 锡尖 | 0.921 | 0.929 |
| 12 | 锡膏 | 0.674 | 0.745 |
