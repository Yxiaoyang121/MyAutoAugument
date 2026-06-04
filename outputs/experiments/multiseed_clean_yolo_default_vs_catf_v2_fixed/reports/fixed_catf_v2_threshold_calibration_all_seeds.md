# Fixed CATF-v2 Threshold Calibration Across All Seeds

No training was run. Calibration is post-hoc and per-class.

## Selected RC Calibration Summary

| seed | selected | P | R | mAP50 | mAP50-95 | Delta P | Delta R | Delta mAP50 | Delta mAP50-95 | constraint_failed |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | catf_v2_rc_threshold_score | 0.6636 | 0.7915 | 0.6737 | 0.4303 | +0.0397 | +0.0269 | +0.0038 | -0.0099 | false |
| 1 | previous_constrained_score | 0.6132 | 0.8295 | 0.7259 | 0.4757 | +0.0460 | +0.0565 | +0.0436 | +0.0312 | false |
| 2 | previous_constrained_score | 0.5916 | 0.8230 | 0.7312 | 0.4711 | -0.0077 | +0.0667 | +0.0452 | +0.0112 | false |

- Constraint pass count after RC calibration: `3/3`.

## Final Recommended Per-Class Threshold Table

| class | threshold | evidence |
|---|---:|---|
| 0:OK2 | 0.25 | {'lower_count': 0, 'raise_count': 1, 'default_count': 2} |
| 1:OK3 | 0.10 | {'lower_count': 3, 'raise_count': 0, 'default_count': 0} |
| 2:加强筋打伤 | 0.25 | {'lower_count': 0, 'raise_count': 1, 'default_count': 2} |
| 3:开裂 | 0.25 | {'lower_count': 0, 'raise_count': 1, 'default_count': 2} |
| 4:油污 | 0.10 | {'lower_count': 2, 'raise_count': 1, 'default_count': 0} |
| 5:浅划伤 | 0.10 | {'lower_count': 3, 'raise_count': 0, 'default_count': 0} |
| 6:漏背锡 | 0.10 | {'lower_count': 3, 'raise_count': 0, 'default_count': 0} |
| 7:碰伤 | 0.10 | {'lower_count': 3, 'raise_count': 0, 'default_count': 0} |
| 8:脏污 | 0.10 | {'lower_count': 2, 'raise_count': 0, 'default_count': 1} |
| 9:轮廓划伤 | 0.10 | {'lower_count': 3, 'raise_count': 0, 'default_count': 0} |
| 10:锡丝残留 | 0.25 | {'lower_count': 1, 'raise_count': 1, 'default_count': 1} |
| 11:锡尖 | 0.25 | {'lower_count': 1, 'raise_count': 0, 'default_count': 2} |
| 12:锡膏 | 0.10 | {'lower_count': 2, 'raise_count': 0, 'default_count': 1} |

## High-Recall-Sensitive Classes

1:OK3, 4:油污, 5:浅划伤, 6:漏背锡, 7:碰伤, 8:脏污, 9:轮廓划伤, 12:锡膏

## Classes Not Fully Fixed By Threshold Alone

2:加强筋打伤, 10:锡丝残留, 9:轮廓划伤, 12:锡膏
