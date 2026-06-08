# CP-CATF Training Validation Plan

No training is executed in this step.

## Plan

- Seeds: `0, 1, 2`
- Compare: clean native YOLO default vs fixed CATF-v2 vs CP-CATF.
- CP-CATF flow: diagnosis -> candidate policy proposal -> causal probe -> accepted policy matrix -> sample router -> ROI augmentation.
- Only candidate policies passing the causal probe may enter image augmentation.
- If no image candidate passes, strict no-op is used; sampler-only may be emitted as pending until dataloader weighting is implemented.
- Evaluate industrial constraints per seed: Precision, mAP50, and mAP50-95 must not drop more than 0.01 vs clean.
- Required outcome for main-method claim: seed0/seed1 retain most fixed CATF-v2 benefit, seed2 is protected, and constraint_failed is 0/3.

## Offline Probe Gate

- Offline probe recommends training validation: `true`.
