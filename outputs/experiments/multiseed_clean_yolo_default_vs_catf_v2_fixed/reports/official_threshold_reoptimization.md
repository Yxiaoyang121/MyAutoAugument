# Official-Path CATF-v2 Threshold Re-optimization

Generated: `2026-06-04T12:54:34`

No training was run. This analysis uses existing Ultralytics `YOLO.predict(conf=0.10)` outputs and the official predict post-processing evaluator.

## Pass Count Summary

| scheme | pass_count | note |
|---|---:|---|
| fixed CATF-v2 without RC | 2/3 | official Ultralytics val constraints |
| old unified RC | 1/3 | saved `catf_v2_rc_per_class_thresholds.json` |
| reoptimized unified RC | 2/3 | one threshold table for all seeds |
| per-seed RC | 3/3 | model/seed-specific deployment calibration |
| conservative default RC | 2/3 | limited changes, high-FP prior protected |

## Reoptimized Unified Thresholds

| class | threshold |
|---|---:|
| 0:OK2 | 0.70 |
| 1:OK3 | 0.70 |
| 2:加强筋打伤 | 0.35 |
| 3:开裂 | 0.50 |
| 4:油污 | 0.50 |
| 5:浅划伤 | 0.10 |
| 6:漏背锡 | 0.10 |
| 7:碰伤 | 0.10 |
| 8:脏污 | 0.10 |
| 9:轮廓划伤 | 0.10 |
| 10:锡丝残留 | 0.25 |
| 11:锡尖 | 0.25 |
| 12:锡膏 | 0.10 |

## Unified Scheme Per-Seed Metrics

| seed | P | R | mAP50 | mAP50-95 | dP | dR | dM50 | dM95 | constraint_failed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | 0.6147 | 0.7907 | 0.6728 | 0.4294 | -0.0093 | +0.0262 | +0.0030 | -0.0108 | true |
| 1 | 0.6573 | 0.7963 | 0.7133 | 0.4706 | +0.0901 | +0.0233 | +0.0310 | +0.0260 | false |
| 2 | 0.6184 | 0.7976 | 0.7223 | 0.4660 | +0.0190 | +0.0413 | +0.0362 | +0.0060 | false |

## Per-Seed Scheme Metrics

| seed | P | R | mAP50 | mAP50-95 | dP | dR | dM50 | dM95 | constraint_failed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | 0.6227 | 0.7967 | 0.6752 | 0.4316 | -0.0012 | +0.0321 | +0.0054 | -0.0086 | false |
| 1 | 0.6583 | 0.8295 | 0.7259 | 0.4757 | +0.0911 | +0.0565 | +0.0436 | +0.0312 | false |
| 2 | 0.6109 | 0.8246 | 0.7322 | 0.4717 | +0.0115 | +0.0684 | +0.0461 | +0.0117 | false |

## Conservative Scheme Metrics

| seed | P | R | mAP50 | mAP50-95 | dP | dR | dM50 | dM95 | constraint_failed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | 0.6113 | 0.7717 | 0.6619 | 0.4242 | -0.0126 | +0.0071 | -0.0079 | -0.0159 | true |
| 1 | 0.6736 | 0.7771 | 0.6989 | 0.4643 | +0.1063 | +0.0041 | +0.0167 | +0.0198 | false |
| 2 | 0.6488 | 0.7872 | 0.7140 | 0.4644 | +0.0494 | +0.0309 | +0.0279 | +0.0044 | false |

## Why Old RC Failed

- The saved RC table lowered many classes to `0.10`, including domain high-FP-prior classes. That recovered Recall but introduced enough false positives to violate the Precision guard on seed0 and seed2.
- Seed0 old RC failed Precision by `-0.0666`; seed2 old RC failed Precision by `-0.0148`.

### Seed0 FP Sources Under Old RC

| class | delta FP | delta Precision |
|---|---:|---:|
| 7:碰伤 | +37.0 | -0.0517 |
| 8:脏污 | +19.0 | -0.1635 |
| 6:漏背锡 | +14.0 | -0.0932 |
| 12:锡膏 | +10.0 | -0.0995 |
| 4:油污 | +10.0 | -0.0748 |
| 9:轮廓划伤 | +10.0 | -0.0440 |
| 5:浅划伤 | +7.0 | -0.2263 |
| 11:锡尖 | +3.0 | -0.0630 |

### Seed2 FP Sources Under Old RC

| class | delta FP | delta Precision |
|---|---:|---:|
| 7:碰伤 | +117.0 | -0.1663 |
| 8:脏污 | +56.0 | -0.3012 |
| 4:油污 | +32.0 | -0.0379 |
| 9:轮廓划伤 | +23.0 | -0.1178 |
| 6:漏背锡 | +22.0 | -0.1759 |
| 12:锡膏 | +19.0 | -0.2352 |
| 1:OK3 | +6.0 | -0.0142 |
| 0:OK2 | +2.0 | -0.0177 |

## Conclusions

- Unified threshold 3/3 possible: `false`.
- Per-seed threshold 3/3 possible: `true`.
- If only per-seed calibration passes, RC should be framed as deployment calibration rather than the core training method.
- Recommended paper line: fixed CATF-v2 is the training method; threshold calibration is a deployment-time calibration layer that must be validated on the official predict path.
