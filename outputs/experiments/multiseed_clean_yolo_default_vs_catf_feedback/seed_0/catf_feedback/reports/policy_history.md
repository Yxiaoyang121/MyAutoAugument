# CATF In-Loop Feedback Policy History

| epoch | action | guards | adjustments | frozen | copy_paste_status |
|---:|---|---|---:|---|---|
| 5 | rollback | precision_guard,low_contrast_fn,recall_low,constraint_warning,rollback | 0 | false | pending_object_bank_design |
| 10 | cooldown | precision_guard,low_contrast_fn | 14 | false | pending_object_bank_design |
| 15 | accept | precision_guard,low_contrast_fn,recall_low | 10 | false | pending_object_bank_design |
| 20 | rollback | precision_guard,low_contrast_fn,constraint_warning,rollback | 10 | false | pending_object_bank_design |
| 25 | cooldown | precision_guard,low_contrast_fn,recall_low | 14 | false | pending_object_bank_design |
| 30 | accept | precision_guard,low_contrast_fn | 7 | false | pending_object_bank_design |
| 35 | accept | precision_guard,low_contrast_fn | 5 | false | pending_object_bank_design |
| 40 | rollback | precision_guard,map50_95_guard,low_contrast_fn,constraint_warning,rollback | 5 | true | pending_object_bank_design |
| 45 | freeze | precision_guard,low_contrast_fn,recall_low | 0 | true | pending_object_bank_design |

## Adjustments

### Epoch 5 - rollback
- Reference metrics: `{'precision': 0.57142, 'recall': 0.47866, 'map50': 0.39571, 'map50_95': 0.23466}`
- Delta metrics: `{'precision': 0.049899999999999944, 'recall': -0.12128999999999995, 'map50': -0.02123999999999998, 'map50_95': -0.0049000000000000155}`
- Last safe policy id: `initial`
- Rollback reason: `map50_drop_lt_0.015`
- Group budget before: `{'photometric': {'used': 0.16, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.03, 'budget': 0.08, 'within_budget': True}}`
- Group budget after: `{'photometric': {'used': 0.16, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.03, 'budget': 0.08, 'within_budget': True}}`
- Trust-region clipping: `[]`
- No adjustment.
### Epoch 10 - cooldown
- Reference metrics: `{'precision': 0.5644, 'recall': 0.43041, 'map50': 0.441, 'map50_95': 0.28921}`
- Delta metrics: `{'precision': 0.03476999999999997, 'recall': 0.07841000000000004, 'map50': 0.10103000000000001, 'map50_95': 0.04543999999999998}`
- Last safe policy id: `initial`
- Rollback reason: `None`
- Group budget before: `{'photometric': {'used': 0.16, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.03, 'budget': 0.08, 'within_budget': True}}`
- Group budget after: `{'photometric': {'used': 0.08, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.15000000000000002, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.009999999999999998, 'budget': 0.08, 'within_budget': True}}`
- Trust-region clipping: `[]`
- `clahe` prob: 0.0500 -> 0.0300 (cooldown_shrink_risk_ops)
- `clahe` strength: 0.2000 -> 0.1700 (cooldown_shrink_risk_ops)
- `gamma` prob: 0.0500 -> 0.0300 (cooldown_shrink_risk_ops)
- `gamma` strength: 0.2000 -> 0.1700 (cooldown_shrink_risk_ops)
- `brightness` prob: 0.0300 -> 0.0100 (cooldown_shrink_risk_ops)
- `brightness` strength: 0.1500 -> 0.1200 (cooldown_shrink_risk_ops)
- `contrast` prob: 0.0300 -> 0.0100 (cooldown_shrink_risk_ops)
- `contrast` strength: 0.1500 -> 0.1200 (cooldown_shrink_risk_ops)
- `sharpen_mild` prob: 0.0800 -> 0.0850 (cooldown_texture_only)
- `sharpen_mild` strength: 0.2500 -> 0.2550 (cooldown_texture_only)
- `local_contrast` prob: 0.0600 -> 0.0650 (cooldown_texture_only)
- `local_contrast` strength: 0.2000 -> 0.2050 (cooldown_texture_only)
- `cutout_safe` prob: 0.0300 -> 0.0100 (cooldown_shrink_risk_ops)
- `cutout_safe` strength: 0.1000 -> 0.0700 (cooldown_shrink_risk_ops)
### Epoch 15 - accept
- Reference metrics: `{'precision': 0.48426, 'recall': 0.61141, 'map50': 0.55296, 'map50_95': 0.35446}`
- Delta metrics: `{'precision': 0.15716999999999992, 'recall': -0.10514000000000001, 'map50': 0.003329999999999944, 'map50_95': 0.007319999999999993}`
- Last safe policy id: `catf_epoch_010_001`
- Rollback reason: `None`
- Group budget before: `{'photometric': {'used': 0.08, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.15000000000000002, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.009999999999999998, 'budget': 0.08, 'within_budget': True}}`
- Group budget after: `{'photometric': {'used': 0.020000000000000004, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.15000000000000002, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.0, 'budget': 0.08, 'within_budget': True}}`
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
### Epoch 20 - rollback
- Reference metrics: `{'precision': 0.65591, 'recall': 0.56434, 'map50': 0.57831, 'map50_95': 0.37719}`
- Delta metrics: `{'precision': -0.07096999999999998, 'recall': 0.020600000000000063, 'map50': 0.0007899999999999574, 'map50_95': 0.0007699999999999929}`
- Last safe policy id: `catf_epoch_010_001`
- Rollback reason: `precision_drop_lt_0.015`
- Group budget before: `{'photometric': {'used': 0.020000000000000004, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.15000000000000002, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.0, 'budget': 0.08, 'within_budget': True}}`
- Group budget after: `{'photometric': {'used': 0.08, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.15000000000000002, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.009999999999999998, 'budget': 0.08, 'within_budget': True}}`
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
### Epoch 25 - cooldown
- Reference metrics: `{'precision': 0.54595, 'recall': 0.64355, 'map50': 0.59895, 'map50_95': 0.39138}`
- Delta metrics: `{'precision': 0.07547999999999999, 'recall': -0.027559999999999918, 'map50': 0.028569999999999984, 'map50_95': 0.011319999999999997}`
- Last safe policy id: `catf_epoch_010_001`
- Rollback reason: `None`
- Group budget before: `{'photometric': {'used': 0.08, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.15000000000000002, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.009999999999999998, 'budget': 0.08, 'within_budget': True}}`
- Group budget after: `{'photometric': {'used': 0.020000000000000004, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.16000000000000003, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.0, 'budget': 0.08, 'within_budget': True}}`
- Trust-region clipping: `[]`
- `clahe` prob: 0.0300 -> 0.0100 (cooldown_shrink_risk_ops)
- `clahe` strength: 0.1700 -> 0.1400 (cooldown_shrink_risk_ops)
- `gamma` prob: 0.0300 -> 0.0100 (cooldown_shrink_risk_ops)
- `gamma` strength: 0.1700 -> 0.1400 (cooldown_shrink_risk_ops)
- `brightness` prob: 0.0100 -> 0.0000 (cooldown_shrink_risk_ops)
- `brightness` strength: 0.1200 -> 0.0900 (cooldown_shrink_risk_ops)
- `contrast` prob: 0.0100 -> 0.0000 (cooldown_shrink_risk_ops)
- `contrast` strength: 0.1200 -> 0.0900 (cooldown_shrink_risk_ops)
- `sharpen_mild` prob: 0.0850 -> 0.0900 (cooldown_texture_only)
- `sharpen_mild` strength: 0.2550 -> 0.2600 (cooldown_texture_only)
- `local_contrast` prob: 0.0650 -> 0.0700 (cooldown_texture_only)
- `local_contrast` strength: 0.2050 -> 0.2100 (cooldown_texture_only)
- `cutout_safe` prob: 0.0100 -> 0.0000 (cooldown_shrink_risk_ops)
- `cutout_safe` strength: 0.0700 -> 0.0400 (cooldown_shrink_risk_ops)
### Epoch 30 - accept
- Reference metrics: `{'precision': 0.62371, 'recall': 0.6195, 'map50': 0.65284, 'map50_95': 0.41607}`
- Delta metrics: `{'precision': -0.0014699999999999713, 'recall': 0.034439999999999915, 'map50': -0.0034199999999999786, 'map50_95': 0.009550000000000003}`
- Last safe policy id: `catf_epoch_025_003`
- Rollback reason: `None`
- Group budget before: `{'photometric': {'used': 0.020000000000000004, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.16000000000000003, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.0, 'budget': 0.08, 'within_budget': True}}`
- Group budget after: `{'photometric': {'used': 0.0, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.16000000000000003, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.0, 'budget': 0.08, 'within_budget': True}}`
- Trust-region clipping: `[]`
- `clahe` prob: 0.0100 -> 0.0000 (precision_guard)
- `clahe` strength: 0.1400 -> 0.1100 (precision_guard)
- `gamma` prob: 0.0100 -> 0.0000 (precision_guard)
- `gamma` strength: 0.1400 -> 0.1100 (precision_guard)
- `brightness` strength: 0.0900 -> 0.0600 (precision_guard)
- `contrast` strength: 0.0900 -> 0.0600 (precision_guard)
- `cutout_safe` strength: 0.0400 -> 0.0100 (precision_guard)
### Epoch 35 - accept
- Reference metrics: `{'precision': 0.62672, 'recall': 0.59279, 'map50': 0.64481, 'map50_95': 0.4186}`
- Delta metrics: `{'precision': 0.01747999999999994, 'recall': 0.12564999999999993, 'map50': 0.07979999999999998, 'map50_95': 0.06755}`
- Last safe policy id: `catf_epoch_030_004`
- Rollback reason: `None`
- Group budget before: `{'photometric': {'used': 0.0, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.16000000000000003, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.0, 'budget': 0.08, 'within_budget': True}}`
- Group budget after: `{'photometric': {'used': 0.0, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.16000000000000003, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.0, 'budget': 0.08, 'within_budget': True}}`
- Trust-region clipping: `[]`
- `clahe` strength: 0.1100 -> 0.0800 (precision_guard)
- `gamma` strength: 0.1100 -> 0.0800 (precision_guard)
- `brightness` strength: 0.0600 -> 0.0300 (precision_guard)
- `contrast` strength: 0.0600 -> 0.0300 (precision_guard)
- `cutout_safe` strength: 0.0100 -> 0.0000 (precision_guard)
### Epoch 40 - rollback
- Reference metrics: `{'precision': 0.67945, 'recall': 0.6025, 'map50': 0.67838, 'map50_95': 0.43599}`
- Delta metrics: `{'precision': -0.043399999999999994, 'recall': 0.0049000000000000155, 'map50': -0.04715999999999998, 'map50_95': -0.023380000000000012}`
- Last safe policy id: `catf_epoch_030_004`
- Rollback reason: `precision_drop_lt_0.015,map50_drop_lt_0.015,map50_95_drop_lt_0.015`
- Group budget before: `{'photometric': {'used': 0.0, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.16000000000000003, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.0, 'budget': 0.08, 'within_budget': True}}`
- Group budget after: `{'photometric': {'used': 0.0, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.16000000000000003, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.0, 'budget': 0.08, 'within_budget': True}}`
- Trust-region clipping: `[]`
- `clahe` strength: 0.0800 -> 0.1100 (catf_update)
- `gamma` strength: 0.0800 -> 0.1100 (catf_update)
- `brightness` strength: 0.0300 -> 0.0600 (catf_update)
- `contrast` strength: 0.0300 -> 0.0600 (catf_update)
- `cutout_safe` strength: 0.0000 -> 0.0100 (catf_update)
### Epoch 45 - freeze
- Reference metrics: `{'precision': 0.66526, 'recall': 0.72701, 'map50': 0.6864, 'map50_95': 0.44696}`
- Delta metrics: `{'precision': 0.07110000000000005, 'recall': -0.05810000000000004, 'map50': 0.06110000000000004, 'map50_95': 0.05826999999999993}`
- Last safe policy id: `catf_epoch_030_004`
- Rollback reason: `None`
- Group budget before: `{'photometric': {'used': 0.0, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.16000000000000003, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.0, 'budget': 0.08, 'within_budget': True}}`
- Group budget after: `{'photometric': {'used': 0.0, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.16000000000000003, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.0, 'budget': 0.08, 'within_budget': True}}`
- Trust-region clipping: `[]`
- No adjustment.
