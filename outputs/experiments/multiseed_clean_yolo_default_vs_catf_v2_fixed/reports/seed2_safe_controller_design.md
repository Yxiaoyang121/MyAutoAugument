# CATF-v2-Safe Seed2 Controller Design

No training was run. This report compares existing clean seed2 and fixed CATF-v2 seed2 curves.

## Curve Localization

- Recall first lags clean by more than 0.015 at epoch: `6`
- mAP50-95 first lags clean by more than 0.008 at epoch: `8`
- First feedback epoch with mAP50-95 guard visible: `10`

## Feedback Context

- Classes activated at epoch 5: `[{'epoch': 5, 'class_id': 9, 'action': 'propose', 'adjustment_count': 4, 'before_status': 'inactive', 'after_status': 'active', 'dominant_issue': 'texture_boundary_weak'}]`
- ROI affected classes over the run: `{'8': 6, '9': 25, '11': 59}`
- shrink/freeze too late: `true`
- No-op at epoch 5 would preserve clean path: `true`
- High-recall baseline protection should trigger: `true`

## Feedback Epoch Deltas

| epoch | dP | dR | dmAP50 | dmAP50-95 |
|---:|---:|---:|---:|---:|
| 5 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| 10 | +0.0368 | -0.0452 | -0.0208 | -0.0181 |
| 15 | -0.0524 | +0.0268 | +0.0055 | -0.0007 |
| 20 | -0.0137 | +0.0013 | -0.0119 | -0.0016 |
| 25 | +0.0496 | -0.0299 | +0.0507 | +0.0524 |
| 30 | -0.0105 | +0.0485 | +0.0476 | +0.0411 |
| 35 | +0.0290 | +0.0279 | +0.0212 | +0.0167 |
| 40 | -0.0141 | -0.0061 | -0.0015 | +0.0052 |
| 45 | +0.0081 | -0.0379 | -0.0306 | -0.0376 |

## Design Conclusion

- Seed2 should trigger Safe mode because clean seed2 is a strong high-Recall/high-mAP baseline and fixed CATF-v2 starts exploring without early Recall/mAP gain.
- The safest intervention is early abstention at epoch 5, before any industrial ROI augmentation can affect subsequent training.
- If CATF-v2 later shows non-active class regression, Safe should enter no-op freeze rather than keep shrinking active classes.
