# Seed0 Post-hoc Per-Class Threshold Calibration

No training was run. This report evaluates seed0 fixed CATF-v2 with per-class confidence thresholds.

## Candidate Summary

| candidate | P | R | mAP50 | mAP50-95 | Delta P | Delta R | Delta mAP50 | Delta mAP50-95 | constraint_failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| fixed_default_0.25 | 0.6290 | 0.7539 | 0.6499 | 0.4207 | +0.0050 | -0.0107 | -0.0200 | -0.0194 | true |
| previous_constrained_score | 0.6148 | 0.7753 | 0.6625 | 0.4244 | -0.0091 | +0.0107 | -0.0073 | -0.0158 | true |
| catf_v2_rc_threshold_score | 0.6636 | 0.7915 | 0.6737 | 0.4303 | +0.0397 | +0.0269 | +0.0038 | -0.0099 | false |

## Selected Calibration

- Selected candidate: `catf_v2_rc_threshold_score`.
- Constraint failed: `false`.
- Metrics: P/R/mAP50/mAP50-95 = `0.6636/0.7915/0.6737/0.4303`.
- Delta vs clean evaluator: `+0.0397/+0.0269/+0.0038/-0.0099`.

## Threshold Changes

| class | old | new | direction |
|---|---:|---:|---|
| 0:OK2 | 0.25 | 0.70 | raise |
| 1:OK3 | 0.25 | 0.10 | lower |
| 2:加强筋打伤 | 0.25 | 0.35 | raise |
| 3:开裂 | 0.25 | 0.50 | raise |
| 4:油污 | 0.25 | 0.70 | raise |
| 5:浅划伤 | 0.25 | 0.15 | lower |
| 6:漏背锡 | 0.25 | 0.10 | lower |
| 7:碰伤 | 0.25 | 0.10 | lower |
| 9:轮廓划伤 | 0.25 | 0.15 | lower |
| 10:锡丝残留 | 0.25 | 0.70 | raise |
| 12:锡膏 | 0.25 | 0.10 | lower |

## Conclusion

- Seed0 can be repaired in the post-hoc evaluator by using an mAP50-95-aware RC threshold objective.
- Naive recall-first threshold lowering is unsafe for seed0 because it can keep mAP50-95 below the constraint.
- This supports deploying threshold calibration as a constrained selector, not as unconditional global threshold lowering.
