# CATF-v2 Fixed Seed1 vs Clean Native Seed1

Run: `outputs/experiments/catf_v2_fixed_seed1_50ep/`

This run was executed after the strict CATF-v2 no-augmentation bypass fix. Unlike the old seed1 CATF-v2 result, this run did apply industrial ROI augmentation, so the metric change is no longer explained by force-skip label/Instances rewrite.

## Metrics

| Run | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Clean native YOLO default seed1 | 0.7725 | 0.6477 | 0.7542 | 0.4799 |
| Old CATF-v2 seed1 | 0.7691 | 0.6950 | 0.7549 | 0.4898 |
| Fixed CATF-v2 seed1 | 0.7852 | 0.7005 | 0.7826 | 0.5189 |

## Delta

| Comparison | ΔPrecision | ΔRecall | ΔmAP50 | ΔmAP50-95 |
|---|---:|---:|---:|---:|
| Fixed CATF-v2 - Clean native | +0.0127 | +0.0528 | +0.0284 | +0.0390 |
| Fixed CATF-v2 - Old CATF-v2 | +0.0161 | +0.0055 | +0.0277 | +0.0291 |

Constraint status relative to clean native seed1: `constraint_failed=false`.

## Augmentation Activity

- Industrial samples augmented: `55`.
- ROI augmentations applied: `56`.
- ROI affected classes: class `11` (`51` times), class `4` (`5` times).
- Industrial ops applied:
  - `local_contrast`: `28`
  - `sharpen_mild`: `27`
- Invalid bbox count: `0`.
- Bbox out-of-bounds count: `0`.
- Class id out-of-bounds count: `0`.

## Activation Behavior

- Active/proposed classes:
  - class `11`, dominant issue `texture_boundary_weak`
  - class `4`, dominant issue `low_recall`
- OK3 was not activated.
- OK3 ROI applied count was `0`.
- Policy actions: `shrink=5`, `freeze=2`, `accept=1`, `observe=1`.

## Interpretation

The fixed seed1 result is better than both clean native seed1 and the old CATF-v2 seed1 result. Because this run recorded actual ROI/industrial augmentation, the improvement is attributable to the active CATF-v2 augmentation path rather than the previously identified no-op label rewrite issue.
