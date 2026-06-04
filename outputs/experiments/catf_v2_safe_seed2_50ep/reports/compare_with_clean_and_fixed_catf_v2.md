# Compare CATF-v2-Safe Seed2 With Clean And Fixed CATF-v2

## Metrics

| Run | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| clean native seed2 | 0.6962 | 0.7286 | 0.7692 | 0.5224 |
| fixed CATF-v2 seed2 | 0.7637 | 0.6863 | 0.7582 | 0.4967 |
| CATF-v2-Safe seed2 | 0.6962 | 0.7286 | 0.7692 | 0.5224 |

## Delta Vs Clean Seed2

| Run | dP | dR | dmAP50 | dmAP50-95 |
|---|---:|---:|---:|---:|
| fixed CATF-v2 | +0.0674 | -0.0423 | -0.0110 | -0.0257 |
| CATF-v2-Safe | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

## Delta Safe Vs Fixed CATF-v2

| dP | dR | dmAP50 | dmAP50-95 |
|---:|---:|---:|---:|
| -0.0674 | +0.0423 | +0.0110 | +0.0257 |

## Safe Controller Outcome

- constraint_failed: `false`
- Safe fallback: `true`
- Trigger reason: `early_abstention_no_recall_or_map_gain` at epoch 5
- Industrial samples augmented: `0`
- ROI applied: `0`
- Router random draws: `0`
- Interpretation: CATF-v2-Safe preserved the strong clean seed2 trajectory and avoided the fixed CATF-v2 conservative failure mode. This is acceptable for seed2 because the goal is not to damage a high-recall baseline.
