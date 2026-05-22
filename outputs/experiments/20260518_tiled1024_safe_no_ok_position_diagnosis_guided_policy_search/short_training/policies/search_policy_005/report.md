# Policy Search Short-Training: search_policy_005

- operations: `scale(p=0.408, s=0.256), gaussian_noise(p=0.264, s=0.127), local_contrast(p=0.415, s=0.202), cutout(p=0.088, s=0.112), contrast(p=0.340, s=0.268), translate(p=0.524, s=0.113)`
- train images / bboxes: `4602` / `6362`
- val images / bboxes: `677` / `905`
- Precision: `0.609`
- Recall: `0.732`
- mAP50: `0.719`
- mAP50-95: `0.478`
- balanced_score: `0.627950`
- recall_priority_score: `0.659500`
- map_priority_score: `0.601900`
- OOM: `false`
- training wall seconds: `2265.1`

## Per-Class Recall/AP50

| class id | class | Recall | AP50 |
| ---: | --- | ---: | ---: |
| 0 | OK2 | 1.000 | 0.995 |
| 1 | OK3 | 0.982 | 0.990 |
| 2 | 加强筋打伤 | 0.875 | 0.884 |
| 3 | 开裂 | 1.000 | 0.995 |
| 4 | 油污 | 0.278 | 0.196 |
| 5 | 浅划伤 | 0.615 | 0.454 |
| 6 | 漏背锡 | 0.739 | 0.669 |
| 7 | 碰伤 | 0.569 | 0.647 |
| 8 | 脏污 | 0.476 | 0.421 |
| 9 | 轮廓划伤 | 0.667 | 0.733 |
| 10 | 锡丝残留 | 0.667 | 0.683 |
| 11 | 锡尖 | 0.963 | 0.949 |
| 12 | 锡膏 | 0.684 | 0.725 |
