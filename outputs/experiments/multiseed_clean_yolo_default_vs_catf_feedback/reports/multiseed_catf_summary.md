# Multiseed Clean YOLO Default vs CATF Feedback

## Setup

- Seeds: `0, 1, 2`
- Model/data: `yolo11n.pt`, `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml`
- Epochs/imgsz/batch/workers/device: `50 / 1024 / 2 / 0 / 0`
- Train images: `2301`
- Fixed augmented dataset generated: `false`
- Clean group: pure native Ultralytics `YOLO.train` with YOLO default augmentation enabled.
- CATF group: single-run in-loop feedback, YOLO default augmentation enabled, industrial online augmentation enabled, per-seed clean native reference curve.
- copy_paste: `pending_object_bank_design`

## Per-Seed Results

| seed | group | Precision | Recall | mAP50 | mAP50-95 |
|---:|---|---:|---:|---:|---:|
| 0 | clean native | 0.7846 | 0.6765 | 0.7347 | 0.4759 |
| 0 | CATF feedback | 0.7574 | 0.6751 | 0.7526 | 0.5204 |
| 1 | clean native | 0.7725 | 0.6477 | 0.7542 | 0.4799 |
| 1 | CATF feedback | 0.7376 | 0.7098 | 0.7550 | 0.5185 |
| 2 | clean native | 0.6962 | 0.7286 | 0.7692 | 0.5224 |
| 2 | CATF feedback | 0.6846 | 0.7166 | 0.7608 | 0.4800 |

## Per-Seed Deltas

| seed | Delta Precision | Delta Recall | Delta mAP50 | Delta mAP50-95 | constraint_failed | CATF verdict | rollback | cooldown | freeze |
|---:|---:|---:|---:|---:|---|---|---:|---:|---:|
| 0 | -0.0272 | -0.0014 | 0.0179 | 0.0445 | `true` | constraint failed | 3 | 2 | 1 |
| 1 | -0.0349 | 0.0621 | 0.0008 | 0.0386 | `true` | constraint failed | 5 | 1 | 2 |
| 2 | -0.0116 | -0.0120 | -0.0084 | -0.0424 | `true` | constraint failed | 9 | 0 | 0 |

## Mean +/- Std

Sample standard deviation across seeds `0/1/2`.

| metric | clean native | CATF feedback | delta |
|---|---:|---:|---:|
| Precision | 0.7511 +/- 0.0479 | 0.7265 +/- 0.0376 | -0.0246 +/- 0.0118 |
| Recall | 0.6843 +/- 0.0410 | 0.7005 +/- 0.0223 | 0.0163 +/- 0.0401 |
| mAP50 | 0.7527 +/- 0.0173 | 0.7561 +/- 0.0042 | 0.0034 +/- 0.0133 |
| mAP50-95 | 0.4927 +/- 0.0258 | 0.5063 +/- 0.0228 | 0.0136 +/- 0.0486 |

## Win Counts

- CATF wins under industrial constraints: `0/3`
- Constraint failed seeds: `3/3`
- All-metric win seeds: `0/3`
- Precision wins: `0/3`
- Recall wins: `1/3`
- mAP50 wins: `2/3`
- mAP50-95 wins: `2/3`

## CATF Control Statistics

- `accept` actions: `4`
- `cooldown` actions: `3`
- `freeze` actions: `3`
- `rollback` actions: `17`
- Total feedback updates: `27`
- Feedback records with guards: `27`
- Feedback records with frozen policy: `15`

## Stability Compared With Old Feedback

- Old in-loop feedback multiseed: constraint failed `2/3`, one seed was a clean win.
- CATF multiseed: constraint failed `3/3`, no seed wins under industrial constraints.
- CATF is visibly more conservative in action logs because rollback/freeze dominate, but the final seed 0/1 Precision drops and seed 2 mAP50-95 drop still violate constraints.

## Conclusion

- Recommendation: `not_paper_main_method`.
- CATF seed42 remains a useful positive case, but seeds 0/1/2 do not validate stability.
- CATF should be reported as an ablation/control-policy attempt rather than the paper main method unless another controller revision passes multiseed constraints.
