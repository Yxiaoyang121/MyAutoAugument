# CATF-v2 Threshold Calibration

- Status: analysis only; training and validation metrics are unchanged.

| class_id | class_name | default | recommended | reason |
|---:|---|---:|---:|---|
| 0 | OK2 | 0.25 | 0.40 | high_fp_raise_threshold |
| 1 | OK3 | 0.25 | 0.40 | high_fp_raise_threshold |
| 2 | 加强筋打伤 | 0.25 | 0.25 | keep_default |
| 3 | 开裂 | 0.25 | 0.40 | high_fp_raise_threshold |
| 4 | 油污 | 0.25 | 0.40 | high_fp_raise_threshold |
| 5 | 浅划伤 | 0.25 | 0.40 | high_fp_raise_threshold |
| 6 | 漏背锡 | 0.25 | 0.40 | high_fp_raise_threshold |
| 7 | 碰伤 | 0.25 | 0.40 | high_fp_raise_threshold |
| 8 | 脏污 | 0.25 | 0.40 | high_fp_raise_threshold |
| 9 | 轮廓划伤 | 0.25 | 0.40 | high_fp_raise_threshold |
| 10 | 锡丝残留 | 0.25 | 0.40 | high_fp_raise_threshold |
| 11 | 锡尖 | 0.25 | 0.15 | low_recall_lower_threshold |
| 12 | 锡膏 | 0.25 | 0.40 | high_fp_raise_threshold |
