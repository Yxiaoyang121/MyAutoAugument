# CATF In-Loop Feedback Policy History

| epoch | action | guards | adjustments | frozen | copy_paste_status |
|---:|---|---|---:|---|---|
| 5 | rollback | precision_guard,map50_95_guard,low_contrast_fn,recall_low,constraint_warning,rollback | 0 | false | pending_object_bank_design |
| 10 | rollback | precision_guard,map50_95_guard,low_contrast_fn,recall_low,constraint_warning,rollback | 0 | true | pending_object_bank_design |
| 15 | rollback | precision_guard,low_contrast_fn,recall_low,constraint_warning,rollback | 0 | true | pending_object_bank_design |
| 20 | rollback | precision_guard,map50_95_guard,low_contrast_fn,constraint_warning,rollback | 0 | true | pending_object_bank_design |
| 25 | rollback | precision_guard,map50_95_guard,low_contrast_fn,recall_low,constraint_warning,rollback | 0 | true | pending_object_bank_design |
| 30 | rollback | precision_guard,map50_95_guard,low_contrast_fn,recall_low,constraint_warning,rollback | 0 | true | pending_object_bank_design |
| 35 | rollback | precision_guard,low_contrast_fn,constraint_warning,rollback | 0 | true | pending_object_bank_design |
| 40 | rollback | precision_guard,map50_95_guard,low_contrast_fn,recall_low,constraint_warning,rollback | 0 | true | pending_object_bank_design |
| 45 | rollback | precision_guard,map50_95_guard,low_contrast_fn,recall_low,constraint_warning,rollback | 0 | true | pending_object_bank_design |

## Adjustments

### Epoch 5 - rollback
- Reference metrics: `{'precision': 0.55691, 'recall': 0.40929, 'map50': 0.36993, 'map50_95': 0.2199}`
- Delta metrics: `{'precision': 0.09999999999999998, 'recall': -0.07129999999999997, 'map50': -0.030299999999999994, 'map50_95': -0.01421}`
- Last safe policy id: `initial`
- Rollback reason: `map50_drop_lt_0.015`
- Group budget before: `{'photometric': {'used': 0.16, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.03, 'budget': 0.08, 'within_budget': True}}`
- Group budget after: `{'photometric': {'used': 0.16, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.03, 'budget': 0.08, 'within_budget': True}}`
- Trust-region clipping: `[]`
- No adjustment.
### Epoch 10 - rollback
- Reference metrics: `{'precision': 0.64056, 'recall': 0.52788, 'map50': 0.53959, 'map50_95': 0.32997}`
- Delta metrics: `{'precision': -0.08686000000000005, 'recall': -0.015669999999999962, 'map50': -0.020469999999999988, 'map50_95': -0.01258999999999999}`
- Last safe policy id: `initial`
- Rollback reason: `precision_drop_lt_0.015,map50_drop_lt_0.015`
- Group budget before: `{'photometric': {'used': 0.16, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.03, 'budget': 0.08, 'within_budget': True}}`
- Group budget after: `{'photometric': {'used': 0.16, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.03, 'budget': 0.08, 'within_budget': True}}`
- Trust-region clipping: `[]`
- No adjustment.
### Epoch 15 - rollback
- Reference metrics: `{'precision': 0.74166, 'recall': 0.51226, 'map50': 0.57096, 'map50_95': 0.3514}`
- Delta metrics: `{'precision': -0.10326000000000002, 'recall': -0.011030000000000095, 'map50': -0.005720000000000058, 'map50_95': -0.0036399999999999766}`
- Last safe policy id: `initial`
- Rollback reason: `precision_drop_lt_0.015`
- Group budget before: `{'photometric': {'used': 0.16, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.03, 'budget': 0.08, 'within_budget': True}}`
- Group budget after: `{'photometric': {'used': 0.16, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.03, 'budget': 0.08, 'within_budget': True}}`
- Trust-region clipping: `[]`
- No adjustment.
### Epoch 20 - rollback
- Reference metrics: `{'precision': 0.71946, 'recall': 0.56093, 'map50': 0.65592, 'map50_95': 0.40923}`
- Delta metrics: `{'precision': -0.22553999999999996, 'recall': 0.08387, 'map50': -0.0793299999999999, 'map50_95': -0.05603999999999998}`
- Last safe policy id: `initial`
- Rollback reason: `precision_drop_lt_0.015,map50_drop_lt_0.015,map50_95_drop_lt_0.015`
- Group budget before: `{'photometric': {'used': 0.16, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.03, 'budget': 0.08, 'within_budget': True}}`
- Group budget after: `{'photometric': {'used': 0.16, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.03, 'budget': 0.08, 'within_budget': True}}`
- Trust-region clipping: `[]`
- No adjustment.
### Epoch 25 - rollback
- Reference metrics: `{'precision': 0.58109, 'recall': 0.72766, 'map50': 0.66478, 'map50_95': 0.40868}`
- Delta metrics: `{'precision': -0.054300000000000015, 'recall': -0.027780000000000027, 'map50': -0.041140000000000065, 'map50_95': -0.03484999999999999}`
- Last safe policy id: `initial`
- Rollback reason: `precision_drop_lt_0.015,map50_drop_lt_0.015,map50_95_drop_lt_0.015`
- Group budget before: `{'photometric': {'used': 0.16, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.03, 'budget': 0.08, 'within_budget': True}}`
- Group budget after: `{'photometric': {'used': 0.16, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.03, 'budget': 0.08, 'within_budget': True}}`
- Trust-region clipping: `[]`
- No adjustment.
### Epoch 30 - rollback
- Reference metrics: `{'precision': 0.6374, 'recall': 0.67374, 'map50': 0.66525, 'map50_95': 0.42486}`
- Delta metrics: `{'precision': -0.023329999999999962, 'recall': -0.0948, 'map50': -0.02039000000000002, 'map50_95': -0.016390000000000016}`
- Last safe policy id: `initial`
- Rollback reason: `precision_drop_lt_0.015,map50_drop_lt_0.015,map50_95_drop_lt_0.015`
- Group budget before: `{'photometric': {'used': 0.16, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.03, 'budget': 0.08, 'within_budget': True}}`
- Group budget after: `{'photometric': {'used': 0.16, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.03, 'budget': 0.08, 'within_budget': True}}`
- Trust-region clipping: `[]`
- No adjustment.
### Epoch 35 - rollback
- Reference metrics: `{'precision': 0.66767, 'recall': 0.64721, 'map50': 0.68598, 'map50_95': 0.44703}`
- Delta metrics: `{'precision': -0.04642999999999997, 'recall': 0.05757000000000001, 'map50': 0.0016999999999999238, 'map50_95': 0.000260000000000038}`
- Last safe policy id: `initial`
- Rollback reason: `precision_drop_lt_0.015`
- Group budget before: `{'photometric': {'used': 0.16, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.03, 'budget': 0.08, 'within_budget': True}}`
- Group budget after: `{'photometric': {'used': 0.16, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.03, 'budget': 0.08, 'within_budget': True}}`
- Trust-region clipping: `[]`
- No adjustment.
### Epoch 40 - rollback
- Reference metrics: `{'precision': 0.70408, 'recall': 0.71413, 'map50': 0.74469, 'map50_95': 0.48639}`
- Delta metrics: `{'precision': -0.07869999999999999, 'recall': -0.05060000000000009, 'map50': -0.05658999999999992, 'map50_95': -0.04997999999999997}`
- Last safe policy id: `initial`
- Rollback reason: `precision_drop_lt_0.015,map50_drop_lt_0.015,map50_95_drop_lt_0.015`
- Group budget before: `{'photometric': {'used': 0.16, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.03, 'budget': 0.08, 'within_budget': True}}`
- Group budget after: `{'photometric': {'used': 0.16, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.03, 'budget': 0.08, 'within_budget': True}}`
- Trust-region clipping: `[]`
- No adjustment.
### Epoch 45 - rollback
- Reference metrics: `{'precision': 0.6907, 'recall': 0.73331, 'map50': 0.77009, 'map50_95': 0.52238}`
- Delta metrics: `{'precision': 0.07957000000000003, 'recall': -0.09548999999999996, 'map50': -0.05190000000000006, 'map50_95': -0.07454999999999995}`
- Last safe policy id: `initial`
- Rollback reason: `map50_drop_lt_0.015,map50_95_drop_lt_0.015`
- Group budget before: `{'photometric': {'used': 0.16, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.03, 'budget': 0.08, 'within_budget': True}}`
- Group budget after: `{'photometric': {'used': 0.16, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.03, 'budget': 0.08, 'within_budget': True}}`
- Trust-region clipping: `[]`
- No adjustment.
