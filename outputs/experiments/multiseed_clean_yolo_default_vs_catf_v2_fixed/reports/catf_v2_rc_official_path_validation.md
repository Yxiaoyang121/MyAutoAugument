# CATF-v2-RC Official Path Validation

Generated: `2026-06-04T12:14:10`

No training was run. Fixed CATF-v2 `best.pt` checkpoints were evaluated with Ultralytics `YOLO.val` and `YOLO.predict`.

Important: per-class thresholds cannot be represented directly in Ultralytics `YOLO.val`, so CATF-v2-RC is evaluated as `YOLO.predict(conf=0.10) + per-class threshold post-processing + the same cached-prediction evaluator used for threshold selection`.

## Per-Class Thresholds

| class | threshold |
|---|---:|
| 0:OK2 | 0.25 |
| 1:OK3 | 0.10 |
| 2:加强筋打伤 | 0.25 |
| 3:开裂 | 0.25 |
| 4:油污 | 0.10 |
| 5:浅划伤 | 0.10 |
| 6:漏背锡 | 0.10 |
| 7:碰伤 | 0.10 |
| 8:脏污 | 0.10 |
| 9:轮廓划伤 | 0.10 |
| 10:锡丝残留 | 0.25 |
| 11:锡尖 | 0.25 |
| 12:锡膏 | 0.10 |

## Official Ultralytics Val: Clean vs Fixed CATF-v2

| seed | clean P | clean R | clean mAP50 | clean mAP50-95 | fixed P | fixed R | fixed mAP50 | fixed mAP50-95 | dP | dR | dM50 | dM95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.7846 | 0.6765 | 0.7347 | 0.4759 | 0.7785 | 0.6697 | 0.7437 | 0.4895 | -0.0060 | -0.0068 | +0.0090 | +0.0136 |
| 1 | 0.7725 | 0.6477 | 0.7542 | 0.4799 | 0.7852 | 0.7005 | 0.7826 | 0.5189 | +0.0127 | +0.0528 | +0.0284 | +0.0390 |
| 2 | 0.6962 | 0.7286 | 0.7692 | 0.5224 | 0.7637 | 0.6863 | 0.7582 | 0.4967 | +0.0674 | -0.0423 | -0.0110 | -0.0257 |

## Official Predict + Post-Processing: Clean vs Fixed vs CATF-v2-RC

| seed | group | P | R | mAP50 | mAP50-95 | dP vs clean | dR vs clean | dM50 vs clean | dM95 vs clean | constraint_failed |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | clean_default_0.25 | 0.6239 | 0.7645 | 0.6698 | 0.4401 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | false |
| 0 | fixed_default_0.25 | 0.6290 | 0.7539 | 0.6499 | 0.4207 | +0.0050 | -0.0107 | -0.0200 | -0.0194 | true |
| 0 | fixed_catf_v2_rc | 0.5573 | 0.7967 | 0.6752 | 0.4307 | -0.0666 | +0.0321 | +0.0054 | -0.0095 | true |
| 1 | clean_default_0.25 | 0.5672 | 0.7730 | 0.6823 | 0.4446 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | false |
| 1 | fixed_default_0.25 | 0.6649 | 0.7647 | 0.6918 | 0.4613 | +0.0977 | -0.0083 | +0.0096 | +0.0168 | false |
| 1 | fixed_catf_v2_rc | 0.6169 | 0.8231 | 0.7218 | 0.4735 | +0.0496 | +0.0501 | +0.0395 | +0.0289 | false |
| 2 | clean_default_0.25 | 0.5994 | 0.7563 | 0.6861 | 0.4599 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | false |
| 2 | fixed_default_0.25 | 0.6438 | 0.7728 | 0.7044 | 0.4611 | +0.0445 | +0.0165 | +0.0183 | +0.0012 | false |
| 2 | fixed_catf_v2_rc | 0.5845 | 0.8218 | 0.7295 | 0.4695 | -0.0148 | +0.0655 | +0.0435 | +0.0096 | true |

## Constraint Summary

- CATF-v2-RC official-predict post-processing pass count: `1/3`.
- RC all seeds pass: `false`.
- Constraint reference for RC is the clean model's `YOLO.predict(conf=0.10)` posthoc default-threshold evaluator for the same seed.
