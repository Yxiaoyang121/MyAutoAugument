# Seed2 Augmentation Operator Attribution

No per-epoch or op-by-class application log exists in the stored stats, so op/class attribution below combines total op counts, ROI affected-class counts, and active policy windows.

## Totals

- Industrial samples augmented: `86`.
- ROI applied: `90`.
- Router random draw count: `5610`.
- ROI affected classes: `{'8': 6, '9': 25, '11': 59}`.

## Operator Counts

| op | seen | applied | skipped probability | skipped ROI unavailable |
|---|---:|---:|---:|---:|
| local_contrast | 2815 | 44 | 2761 | 10 |
| sharpen_mild | 2815 | 42 | 2763 | 10 |

## Policy Windows

| class | window | inferred ops | final dR | final dAP50 | final dAP95 |
|---|---|---|---:|---:|---:|
| 9:轮廓划伤 | 5->10 | `sharpen_mild`, `local_contrast` | -0.1667 | -0.0466 | -0.0654 |
| 11:锡尖 | 25->40 | `sharpen_mild`, `local_contrast` | +0.0046 | +0.0042 | +0.0133 |
| 8:脏污 | 35->40 | `sharpen_mild`, `local_contrast` | +0.0301 | -0.0559 | -0.0502 |

## Attribution Judgment

- The first degradation window occurs immediately after epoch 5, when class 9 receives `sharpen_mild` and `local_contrast`.
- The stored data cannot separate `sharpen_mild` from `local_contrast`, because both were enabled together for every active class.
- Texture ROI ops are the most suspicious operator family for seed2, especially for class 9; class 11 improved, so the operator is not uniformly harmful.
- A per-op short ablation is required before banning one op globally.
