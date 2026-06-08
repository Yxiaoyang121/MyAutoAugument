# CP-CATF seed comparison

- Seed: `1`
- Selected candidate: `candidate_policy_1_roi_texture` / `accept`
- Image modification allowed: `true`
- Probe rejected image augmentation: `false`
- Constraint failed: `false`
- Failure reasons: `[]`

| Group | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| clean native | 0.7725 | 0.6477 | 0.7542 | 0.4799 |
| fixed CATF-v2 | 0.7852 | 0.7005 | 0.7826 | 0.5189 |
| CP-CATF | 0.7852 | 0.7005 | 0.7826 | 0.5189 |

| Delta | dP | dR | d mAP50 | d mAP50-95 |
|---|---:|---:|---:|---:|
| CP-CATF vs clean | +0.0127 | +0.0528 | +0.0284 | +0.0390 |
| CP-CATF vs fixed | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

- Industrial image samples augmented: `55`
- ROI applied: `56`
- Router random draw count: `4260`
- ROI affected classes: `{'4': 5, '11': 51}`
- OK3 active: `false`
- OK3 ROI applied: `0`
- Epoch integrity: `{'epoch_count': None, 'first_epoch': None, 'last_epoch': None, 'epoch_continuous': True}`

Development-mode note: offline probe decisions use existing validation diagnostics for this engineering validation; paper-mode CP-CATF needs a train/probe split or train hard-example probe set.
