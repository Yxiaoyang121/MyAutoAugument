# Fixed CATF-v2 Next Step Summary

- Fixed CATF-v2 current constraint pass count: `2/3`.
- After post-hoc constrained threshold calibration: `2/3`.
- Seed2 repaired by threshold calibration: `True`.

## Conclusion

Seed2 can be repaired by per-class threshold calibration in the post-hoc evaluator, but the calibrated pass count remains below 3/3 because another seed still violates the mAP50-95 constraint.
Fixed CATF-v2 should be treated as a strong candidate rather than a finalized stable论文主方法.
The next implementation work should prioritize class-level rollback, negative-effect attribution, high-recall baseline protection, and a more conservative calibration objective for the remaining failing seed.

## Recommended Paper Wording

- Safe statement: `CATF-v2 improves constraint pass rate and removes no-op augmentation artifacts; remaining instability is isolated to a high-recall seed and motivates threshold calibration / class-level rollback.`
- Avoid claiming `stable superiority over YOLO default` unless all calibrated seeds pass under the same evaluator and official validation protocol.
