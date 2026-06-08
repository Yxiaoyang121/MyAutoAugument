# CP-CATF seed comparison

- Seed: `2`
- Selected candidate: `candidate_policy_3_sampler_only` / `sampler_only`
- Image modification allowed: `false`
- Probe rejected image augmentation: `true`
- Constraint failed: `false`
- Failure reasons: `[]`

| Group | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| clean native | 0.6962 | 0.7286 | 0.7692 | 0.5224 |
| fixed CATF-v2 | 0.7637 | 0.6863 | 0.7582 | 0.4967 |
| CP-CATF | 0.6962 | 0.7286 | 0.7692 | 0.5224 |

| Delta | dP | dR | d mAP50 | d mAP50-95 |
|---|---:|---:|---:|---:|
| CP-CATF vs clean | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| CP-CATF vs fixed | -0.0674 | +0.0423 | +0.0110 | +0.0257 |

- Industrial image samples augmented: `0`
- ROI applied: `0`
- Router random draw count: `0`
- ROI affected classes: `{}`
- OK3 active: `false`
- OK3 ROI applied: `0`
- Epoch integrity: `{'epoch_count': None, 'first_epoch': None, 'last_epoch': None, 'epoch_continuous': True}`

Development-mode note: offline probe decisions use existing validation diagnostics for this engineering validation; paper-mode CP-CATF needs a train/probe split or train hard-example probe set.
