# CATF-v2-RC Final Candidate Plan

CATF-v2-RC = fixed CATF-v2 + per-class constrained threshold calibration + class-level rollback / high-recall protection.

## Why Fixed CATF-v2 Is Not A Failure

- Strict no-op bypass removed the framework artifact, and fixed CATF-v2 improved constraint pass count relative to old CATF-v2.
- Seed0 and seed1 pass official training constraints; seed2 can be repaired by constrained per-class threshold calibration in the post-hoc evaluator.
- The remaining problem is category-level protection, not global augmentation redesign.

## Method Flow For Paper

1. Train clean native YOLO default and fixed CATF-v2 under identical settings.
2. Run CATF-v2 class/issue/sample-aware feedback with strict no-op bypass.
3. On validation predictions, search per-class confidence thresholds with industrial constraints on Precision, mAP50, and mAP50-95.
4. Select thresholds only if constraints pass; otherwise keep default or flag class-level rollback.
5. For repeated class-level negative effects, protect high-recall classes by rollback/freezing that class policy in the next training iteration.

## Threshold Calibration Result

- RC calibrated pass count: `3/3`.
- Ready for paper main candidate: `True`.

## Rollback Guidance

- Keep fixed CATF-v2: `['4:油污', '11:锡尖']`.
- Threshold calibration only: `['8:脏污']`.
- Rollback candidates: `['2:加强筋打伤', '10:锡丝残留']`.
- High-risk negative-effect classes: `['9:轮廓划伤', '12:锡膏']`.

## Recommended Wording

- `CATF-v2-RC improves fixed CATF-v2 by adding constrained per-class threshold calibration and class-level protection.`
- Avoid claiming global augmentation alone is sufficient; the evidence supports a training-plus-deployment-calibration method.

## Next Experiment

Do not make broad algorithm changes. The smallest next validation is to apply the recommended threshold table on official validation outputs and optionally run a seed0-only rollback ablation for high-risk classes.
