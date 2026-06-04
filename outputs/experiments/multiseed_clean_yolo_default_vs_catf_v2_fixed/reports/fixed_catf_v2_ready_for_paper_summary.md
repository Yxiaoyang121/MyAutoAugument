# Fixed CATF-v2 Ready-For-Paper Summary

This summary uses the RC post-hoc threshold selector. No training was run.

## Final Groups

- `clean native`: original per-seed YOLO default baselines.
- `fixed CATF-v2`: strict no-op bypass fixed training results.
- `fixed CATF-v2 + RC calibration`: fixed CATF-v2 predictions with constrained per-class threshold calibration.

RC calibrated constraint pass count: `3/3`.

## Seed-Level Metrics

Official rows are Ultralytics validation metrics. Posthoc rows are from the cached prediction JSON threshold evaluator and should be used for threshold selection, not as a direct replacement for official validation.

| seed | evaluator | group | Precision | Recall | mAP50 | mAP50-95 |
|---:|---|---|---:|---:|---:|---:|
| 0 | official | official_clean_native | 0.7846 | 0.6765 | 0.7347 | 0.4759 |
| 0 | official | official_fixed_catf_v2 | 0.7785 | 0.6697 | 0.7437 | 0.4895 |
| 0 | posthoc | posthoc_clean_default_0.25 | 0.6239 | 0.7645 | 0.6698 | 0.4401 |
| 0 | posthoc | posthoc_fixed_default_0.25 | 0.6290 | 0.7539 | 0.6499 | 0.4207 |
| 0 | posthoc | posthoc_fixed_catf_v2_rc_calibration | 0.6636 | 0.7915 | 0.6737 | 0.4303 |
| 1 | official | official_clean_native | 0.7725 | 0.6477 | 0.7542 | 0.4799 |
| 1 | official | official_fixed_catf_v2 | 0.7852 | 0.7005 | 0.7826 | 0.5189 |
| 1 | posthoc | posthoc_clean_default_0.25 | 0.5672 | 0.7730 | 0.6823 | 0.4446 |
| 1 | posthoc | posthoc_fixed_default_0.25 | 0.6649 | 0.7647 | 0.6918 | 0.4613 |
| 1 | posthoc | posthoc_fixed_catf_v2_rc_calibration | 0.6132 | 0.8295 | 0.7259 | 0.4757 |
| 2 | official | official_clean_native | 0.6962 | 0.7286 | 0.7692 | 0.5224 |
| 2 | official | official_fixed_catf_v2 | 0.7637 | 0.6863 | 0.7582 | 0.4967 |
| 2 | posthoc | posthoc_clean_default_0.25 | 0.5994 | 0.7563 | 0.6861 | 0.4599 |
| 2 | posthoc | posthoc_fixed_default_0.25 | 0.6438 | 0.7728 | 0.7044 | 0.4611 |
| 2 | posthoc | posthoc_fixed_catf_v2_rc_calibration | 0.5916 | 0.8230 | 0.7312 | 0.4711 |

## Recommended Per-Class Thresholds

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

## Main Conclusion

Fixed CATF-v2 becomes a paper-ready candidate when paired with constrained per-class threshold calibration and class-level rollback/high-recall protection as the RC variant.

## Risks

- The threshold evaluator is post-hoc and should be confirmed against the official validation/export pipeline.
- Low-support classes remain noisy; rollback decisions should be treated as protective rather than proof of augmentation harm.
