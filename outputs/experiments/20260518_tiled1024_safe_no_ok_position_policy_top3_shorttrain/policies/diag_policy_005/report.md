# Short-Training Report: diag_policy_005

- policy_id: `diag_policy_005`
- source_issue: `combined`
- operations: `copy_paste(p=0.717, s=0.696), clahe(p=0.459, s=0.422), contrast(p=0.464, s=0.422), gamma(p=0.447, s=0.402), scale(p=0.344, s=0.251), gaussian_noise(p=0.167, s=0.120)`
- contains_copy_paste: `true`
- train images / bboxes: `4602` / `9266`
- val images / bboxes: `677` / `905`
- Precision: `0.714`
- Recall: `0.583`
- mAP50: `0.683`
- mAP50-95: `0.453`
- small_object_recall: `0.6587436332767402`
- short_train_score: `0.578862`
- score formula: `0.30*mAP50 + 0.35*mAP50-95 + 0.20*Recall + 0.15*small_object_recall`
- OOM: `false`
- training wall seconds: `2214.7`

## Per-Class Recall/AP50

| class id | class | Recall | AP50 |
| ---: | --- | ---: | ---: |
| 0 | OK2 | 1.000 | 0.995 |
| 1 | OK3 | 0.989 | 0.976 |
| 2 | 加强筋打伤 | 0.740 | 0.761 |
| 3 | 开裂 | 0.000 | 0.497 |
| 4 | 油污 | 0.222 | 0.155 |
| 5 | 浅划伤 | 0.615 | 0.664 |
| 6 | 漏背锡 | 0.601 | 0.647 |
| 7 | 碰伤 | 0.405 | 0.580 |
| 8 | 脏污 | 0.429 | 0.481 |
| 9 | 轮廓划伤 | 0.533 | 0.714 |
| 10 | 锡丝残留 | 0.419 | 0.726 |
| 11 | 锡尖 | 0.926 | 0.946 |
| 12 | 锡膏 | 0.696 | 0.738 |
