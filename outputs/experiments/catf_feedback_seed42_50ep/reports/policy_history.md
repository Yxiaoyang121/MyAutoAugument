# CATF In-Loop Feedback Policy History

| epoch | action | guards | adjustments | frozen | copy_paste_status |
|---:|---|---|---:|---|---|
| 5 | accept | precision_guard,low_contrast_fn | 10 | false | pending_object_bank_design |
| 10 | accept | precision_guard,low_contrast_fn,recall_low | 10 | false | pending_object_bank_design |
| 15 | rollback | precision_guard,map50_95_guard,low_contrast_fn,constraint_warning,rollback | 10 | false | pending_object_bank_design |
| 20 | rollback | precision_guard,map50_95_guard,low_contrast_fn,constraint_warning,rollback | 0 | true | pending_object_bank_design |
| 25 | shrink | precision_guard,map50_95_guard,low_contrast_fn,recall_low,constraint_warning | 0 | true | pending_object_bank_design |
| 30 | accept | precision_guard,low_contrast_fn | 0 | true | pending_object_bank_design |
| 35 | rollback | precision_guard,low_contrast_fn,constraint_warning,rollback | 0 | true | pending_object_bank_design |
| 40 | rollback | precision_guard,low_contrast_fn,constraint_warning,rollback | 0 | true | pending_object_bank_design |
| 45 | shrink | precision_guard,map50_95_guard,low_contrast_fn,constraint_warning | 0 | true | pending_object_bank_design |

## Adjustments

### Epoch 5 - accept
- Reference metrics: `{'precision': 0.62773, 'recall': 0.36765, 'map50': 0.3489, 'map50_95': 0.21083}`
- Delta metrics: `{'precision': 0.04954000000000003, 'recall': -0.006389999999999951, 'map50': 0.026550000000000018, 'map50_95': 0.002799999999999997}`
- Last safe policy id: `initial`
- Rollback reason: `None`
- Group budget before: `{'photometric': {'used': 0.16, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.03, 'budget': 0.08, 'within_budget': True}}`
- Group budget after: `{'photometric': {'used': 0.08, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.009999999999999998, 'budget': 0.08, 'within_budget': True}}`
- Trust-region clipping: `[]`
- `clahe` prob: 0.0500 -> 0.0300 (precision_guard)
- `clahe` strength: 0.2000 -> 0.1700 (precision_guard)
- `gamma` prob: 0.0500 -> 0.0300 (precision_guard)
- `gamma` strength: 0.2000 -> 0.1700 (precision_guard)
- `brightness` prob: 0.0300 -> 0.0100 (precision_guard)
- `brightness` strength: 0.1500 -> 0.1200 (precision_guard)
- `contrast` prob: 0.0300 -> 0.0100 (precision_guard)
- `contrast` strength: 0.1500 -> 0.1200 (precision_guard)
- `cutout_safe` prob: 0.0300 -> 0.0100 (precision_guard)
- `cutout_safe` strength: 0.1000 -> 0.0700 (precision_guard)
### Epoch 10 - accept
- Reference metrics: `{'precision': 0.54715, 'recall': 0.46071, 'map50': 0.4407, 'map50_95': 0.27563}`
- Delta metrics: `{'precision': 0.061759999999999926, 'recall': -0.05103000000000002, 'map50': -0.006239999999999968, 'map50_95': -0.00928000000000001}`
- Last safe policy id: `catf_epoch_005_001`
- Rollback reason: `None`
- Group budget before: `{'photometric': {'used': 0.08, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.009999999999999998, 'budget': 0.08, 'within_budget': True}}`
- Group budget after: `{'photometric': {'used': 0.020000000000000004, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.0, 'budget': 0.08, 'within_budget': True}}`
- Trust-region clipping: `[]`
- `clahe` prob: 0.0300 -> 0.0100 (precision_guard)
- `clahe` strength: 0.1700 -> 0.1400 (precision_guard)
- `gamma` prob: 0.0300 -> 0.0100 (precision_guard)
- `gamma` strength: 0.1700 -> 0.1400 (precision_guard)
- `brightness` prob: 0.0100 -> 0.0000 (precision_guard)
- `brightness` strength: 0.1200 -> 0.0900 (precision_guard)
- `contrast` prob: 0.0100 -> 0.0000 (precision_guard)
- `contrast` strength: 0.1200 -> 0.0900 (precision_guard)
- `cutout_safe` prob: 0.0100 -> 0.0000 (precision_guard)
- `cutout_safe` strength: 0.0700 -> 0.0400 (precision_guard)
### Epoch 15 - rollback
- Reference metrics: `{'precision': 0.65857, 'recall': 0.4871, 'map50': 0.54112, 'map50_95': 0.34617}`
- Delta metrics: `{'precision': -0.04498000000000002, 'recall': 0.06691999999999998, 'map50': -0.010210000000000052, 'map50_95': -0.011679999999999968}`
- Last safe policy id: `catf_epoch_005_001`
- Rollback reason: `precision_drop_lt_0.015`
- Group budget before: `{'photometric': {'used': 0.020000000000000004, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.0, 'budget': 0.08, 'within_budget': True}}`
- Group budget after: `{'photometric': {'used': 0.08, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.009999999999999998, 'budget': 0.08, 'within_budget': True}}`
- Trust-region clipping: `[]`
- `clahe` prob: 0.0100 -> 0.0300 (catf_update)
- `clahe` strength: 0.1400 -> 0.1700 (catf_update)
- `gamma` prob: 0.0100 -> 0.0300 (catf_update)
- `gamma` strength: 0.1400 -> 0.1700 (catf_update)
- `brightness` prob: 0.0000 -> 0.0100 (catf_update)
- `brightness` strength: 0.0900 -> 0.1200 (catf_update)
- `contrast` prob: 0.0000 -> 0.0100 (catf_update)
- `contrast` strength: 0.0900 -> 0.1200 (catf_update)
- `cutout_safe` prob: 0.0000 -> 0.0100 (catf_update)
- `cutout_safe` strength: 0.0400 -> 0.0700 (catf_update)
### Epoch 20 - rollback
- Reference metrics: `{'precision': 0.59857, 'recall': 0.55488, 'map50': 0.58275, 'map50_95': 0.37716}`
- Delta metrics: `{'precision': -0.01589000000000007, 'recall': 0.05202999999999991, 'map50': -0.015179999999999971, 'map50_95': -0.022199999999999998}`
- Last safe policy id: `catf_epoch_005_001`
- Rollback reason: `precision_drop_lt_0.015,map50_drop_lt_0.015,map50_95_drop_lt_0.015`
- Group budget before: `{'photometric': {'used': 0.08, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.009999999999999998, 'budget': 0.08, 'within_budget': True}}`
- Group budget after: `{'photometric': {'used': 0.08, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.009999999999999998, 'budget': 0.08, 'within_budget': True}}`
- Trust-region clipping: `[]`
- No adjustment.
### Epoch 25 - shrink
- Reference metrics: `{'precision': 0.60194, 'recall': 0.61672, 'map50': 0.61997, 'map50_95': 0.39002}`
- Delta metrics: `{'precision': 0.022699999999999942, 'recall': -0.05649000000000004, 'map50': -0.013680000000000025, 'map50_95': -0.012229999999999963}`
- Last safe policy id: `catf_epoch_005_001`
- Rollback reason: `None`
- Group budget before: `{'photometric': {'used': 0.08, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.009999999999999998, 'budget': 0.08, 'within_budget': True}}`
- Group budget after: `{'photometric': {'used': 0.08, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.009999999999999998, 'budget': 0.08, 'within_budget': True}}`
- Trust-region clipping: `[]`
- No adjustment.
### Epoch 30 - accept
- Reference metrics: `{'precision': 0.63905, 'recall': 0.64703, 'map50': 0.65118, 'map50_95': 0.42492}`
- Delta metrics: `{'precision': -0.0038799999999999946, 'recall': 0.08609, 'map50': 0.04798000000000002, 'map50_95': 0.03586999999999996}`
- Last safe policy id: `catf_epoch_005_001`
- Rollback reason: `None`
- Group budget before: `{'photometric': {'used': 0.08, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.009999999999999998, 'budget': 0.08, 'within_budget': True}}`
- Group budget after: `{'photometric': {'used': 0.08, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.009999999999999998, 'budget': 0.08, 'within_budget': True}}`
- Trust-region clipping: `[]`
- No adjustment.
### Epoch 35 - rollback
- Reference metrics: `{'precision': 0.75123, 'recall': 0.61119, 'map50': 0.66373, 'map50_95': 0.43627}`
- Delta metrics: `{'precision': -0.029169999999999918, 'recall': 0.053810000000000024, 'map50': 0.03747999999999996, 'map50_95': 0.03660000000000002}`
- Last safe policy id: `catf_epoch_005_001`
- Rollback reason: `precision_drop_lt_0.015`
- Group budget before: `{'photometric': {'used': 0.08, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.009999999999999998, 'budget': 0.08, 'within_budget': True}}`
- Group budget after: `{'photometric': {'used': 0.08, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.009999999999999998, 'budget': 0.08, 'within_budget': True}}`
- Trust-region clipping: `[]`
- No adjustment.
### Epoch 40 - rollback
- Reference metrics: `{'precision': 0.69766, 'recall': 0.7267, 'map50': 0.72532, 'map50_95': 0.48443}`
- Delta metrics: `{'precision': -0.024679999999999924, 'recall': 0.0038399999999999546, 'map50': 0.021320000000000006, 'map50_95': 0.01034999999999997}`
- Last safe policy id: `catf_epoch_005_001`
- Rollback reason: `precision_drop_lt_0.015`
- Group budget before: `{'photometric': {'used': 0.08, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.009999999999999998, 'budget': 0.08, 'within_budget': True}}`
- Group budget after: `{'photometric': {'used': 0.08, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.009999999999999998, 'budget': 0.08, 'within_budget': True}}`
- Trust-region clipping: `[]`
- No adjustment.
### Epoch 45 - shrink
- Reference metrics: `{'precision': 0.72549, 'recall': 0.69329, 'map50': 0.75179, 'map50_95': 0.51896}`
- Delta metrics: `{'precision': 0.04808000000000001, 'recall': 0.023660000000000014, 'map50': 0.007520000000000082, 'map50_95': -0.014189999999999925}`
- Last safe policy id: `catf_epoch_005_001`
- Rollback reason: `None`
- Group budget before: `{'photometric': {'used': 0.08, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.009999999999999998, 'budget': 0.08, 'within_budget': True}}`
- Group budget after: `{'photometric': {'used': 0.08, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.009999999999999998, 'budget': 0.08, 'within_budget': True}}`
- Trust-region clipping: `[]`
- No adjustment.
