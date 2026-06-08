# CP-CATF seed comparison

- Seed: `0`
- Selected candidate: `candidate_policy_1_roi_texture` / `accept`
- Image modification allowed: `true`
- Probe rejected image augmentation: `false`
- Constraint failed: `false`
- Failure reasons: `[]`

| Group | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| clean native | 0.7846 | 0.6765 | 0.7347 | 0.4759 |
| fixed CATF-v2 | 0.7785 | 0.6697 | 0.7437 | 0.4895 |
| CP-CATF | 0.7785 | 0.6697 | 0.7437 | 0.4895 |

| Delta | dP | dR | d mAP50 | d mAP50-95 |
|---|---:|---:|---:|---:|
| CP-CATF vs clean | -0.0060 | -0.0068 | +0.0090 | +0.0136 |
| CP-CATF vs fixed | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

- Industrial image samples augmented: `41`
- ROI applied: `45`
- Router random draw count: `3050`
- ROI affected classes: `{'4': 9, '11': 22, '12': 14}`
- OK3 active: `false`
- OK3 ROI applied: `0`
- Epoch integrity: `{'epoch_count': None, 'first_epoch': None, 'last_epoch': None, 'epoch_continuous': True}`

Development-mode note: offline probe decisions use existing validation diagnostics for this engineering validation; paper-mode CP-CATF needs a train/probe split or train hard-example probe set.
