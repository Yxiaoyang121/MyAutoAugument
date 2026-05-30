# CATF In-Loop Feedback Policy History

| epoch | action | guards | adjustments | frozen | copy_paste_status |
|---:|---|---|---:|---|---|
| 5 | accept | precision_guard,low_contrast_fn,recall_low | 10 | false | pending_object_bank_design |
| 10 | rollback | precision_guard,map50_95_guard,low_contrast_fn,recall_low,constraint_warning,rollback | 10 | false | pending_object_bank_design |
| 15 | cooldown | precision_guard,low_contrast_fn | 14 | false | pending_object_bank_design |
| 20 | rollback | precision_guard,map50_95_guard,low_contrast_fn,constraint_warning,rollback | 14 | false | pending_object_bank_design |
| 25 | rollback | precision_guard,low_contrast_fn,constraint_warning,rollback | 0 | true | pending_object_bank_design |
| 30 | rollback | precision_guard,map50_95_guard,low_contrast_fn,constraint_warning,rollback | 0 | true | pending_object_bank_design |
| 35 | rollback | precision_guard,map50_95_guard,low_contrast_fn,constraint_warning,rollback | 0 | true | pending_object_bank_design |
| 40 | freeze | precision_guard,low_contrast_fn | 0 | true | pending_object_bank_design |
| 45 | freeze | precision_guard,low_contrast_fn,recall_low | 0 | true | pending_object_bank_design |

## Adjustments

### Epoch 5 - accept
- Reference metrics: `{'precision': 0.52472, 'recall': 0.38057, 'map50': 0.35714, 'map50_95': 0.21955}`
- Delta metrics: `{'precision': 0.04583999999999999, 'recall': -0.03437000000000001, 'map50': 0.013000000000000012, 'map50_95': 0.005530000000000007}`
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
### Epoch 10 - rollback
- Reference metrics: `{'precision': 0.58297, 'recall': 0.529, 'map50': 0.50905, 'map50_95': 0.30616}`
- Delta metrics: `{'precision': -0.053590000000000027, 'recall': -0.020930000000000004, 'map50': -0.056999999999999995, 'map50_95': -0.03051999999999999}`
- Last safe policy id: `initial`
- Rollback reason: `precision_drop_lt_0.015,map50_drop_lt_0.015,map50_95_drop_lt_0.015`
- Group budget before: `{'photometric': {'used': 0.08, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.009999999999999998, 'budget': 0.08, 'within_budget': True}}`
- Group budget after: `{'photometric': {'used': 0.16, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.03, 'budget': 0.08, 'within_budget': True}}`
- Trust-region clipping: `[]`
- `clahe` prob: 0.0300 -> 0.0500 (catf_update)
- `clahe` strength: 0.1700 -> 0.2000 (catf_update)
- `gamma` prob: 0.0300 -> 0.0500 (catf_update)
- `gamma` strength: 0.1700 -> 0.2000 (catf_update)
- `brightness` prob: 0.0100 -> 0.0300 (catf_update)
- `brightness` strength: 0.1200 -> 0.1500 (catf_update)
- `contrast` prob: 0.0100 -> 0.0300 (catf_update)
- `contrast` strength: 0.1200 -> 0.1500 (catf_update)
- `cutout_safe` prob: 0.0100 -> 0.0300 (catf_update)
- `cutout_safe` strength: 0.0700 -> 0.1000 (catf_update)
### Epoch 15 - cooldown
- Reference metrics: `{'precision': 0.58403, 'recall': 0.53691, 'map50': 0.55771, 'map50_95': 0.34161}`
- Delta metrics: `{'precision': 0.02864, 'recall': -0.004210000000000047, 'map50': 0.00039999999999995595, 'map50_95': 0.012369999999999992}`
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
### Epoch 20 - rollback
- Reference metrics: `{'precision': 0.61505, 'recall': 0.62255, 'map50': 0.60204, 'map50_95': 0.38584}`
- Delta metrics: `{'precision': 0.008179999999999965, 'recall': 0.0019299999999999873, 'map50': -0.0007400000000000739, 'map50_95': -0.01586000000000004}`
- Last safe policy id: `initial`
- Rollback reason: `map50_95_drop_lt_0.015`
- Group budget before: `{'photometric': {'used': 0.08, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.15000000000000002, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.009999999999999998, 'budget': 0.08, 'within_budget': True}}`
- Group budget after: `{'photometric': {'used': 0.16, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.03, 'budget': 0.08, 'within_budget': True}}`
- Trust-region clipping: `[]`
- `clahe` prob: 0.0300 -> 0.0500 (catf_update)
- `clahe` strength: 0.1700 -> 0.2000 (catf_update)
- `gamma` prob: 0.0300 -> 0.0500 (catf_update)
- `gamma` strength: 0.1700 -> 0.2000 (catf_update)
- `brightness` prob: 0.0100 -> 0.0300 (catf_update)
- `brightness` strength: 0.1200 -> 0.1500 (catf_update)
- `contrast` prob: 0.0100 -> 0.0300 (catf_update)
- `contrast` strength: 0.1200 -> 0.1500 (catf_update)
- `sharpen_mild` prob: 0.0850 -> 0.0800 (catf_update)
- `sharpen_mild` strength: 0.2550 -> 0.2500 (catf_update)
- `local_contrast` prob: 0.0650 -> 0.0600 (catf_update)
- `local_contrast` strength: 0.2050 -> 0.2000 (catf_update)
- `cutout_safe` prob: 0.0100 -> 0.0300 (catf_update)
- `cutout_safe` strength: 0.0700 -> 0.1000 (catf_update)
### Epoch 25 - rollback
- Reference metrics: `{'precision': 0.71541, 'recall': 0.55099, 'map50': 0.61058, 'map50_95': 0.3851}`
- Delta metrics: `{'precision': -0.12244, 'recall': 0.16437000000000002, 'map50': 0.07494999999999996, 'map50_95': 0.05141000000000001}`
- Last safe policy id: `initial`
- Rollback reason: `precision_drop_lt_0.015`
- Group budget before: `{'photometric': {'used': 0.16, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.03, 'budget': 0.08, 'within_budget': True}}`
- Group budget after: `{'photometric': {'used': 0.16, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.03, 'budget': 0.08, 'within_budget': True}}`
- Trust-region clipping: `[]`
- No adjustment.
### Epoch 30 - rollback
- Reference metrics: `{'precision': 0.72395, 'recall': 0.64418, 'map50': 0.71701, 'map50_95': 0.46425}`
- Delta metrics: `{'precision': -0.17142000000000002, 'recall': 0.02690999999999999, 'map50': -0.05335000000000001, 'map50_95': -0.030090000000000006}`
- Last safe policy id: `initial`
- Rollback reason: `precision_drop_lt_0.015,map50_drop_lt_0.015,map50_95_drop_lt_0.015`
- Group budget before: `{'photometric': {'used': 0.16, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.03, 'budget': 0.08, 'within_budget': True}}`
- Group budget after: `{'photometric': {'used': 0.16, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.03, 'budget': 0.08, 'within_budget': True}}`
- Trust-region clipping: `[]`
- No adjustment.
### Epoch 35 - rollback
- Reference metrics: `{'precision': 0.75408, 'recall': 0.64917, 'map50': 0.71952, 'map50_95': 0.46908}`
- Delta metrics: `{'precision': -0.18162, 'recall': 0.09189000000000003, 'map50': -0.06452000000000002, 'map50_95': -0.048929999999999974}`
- Last safe policy id: `initial`
- Rollback reason: `precision_drop_lt_0.015,map50_drop_lt_0.015,map50_95_drop_lt_0.015`
- Group budget before: `{'photometric': {'used': 0.16, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.03, 'budget': 0.08, 'within_budget': True}}`
- Group budget after: `{'photometric': {'used': 0.16, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.03, 'budget': 0.08, 'within_budget': True}}`
- Trust-region clipping: `[]`
- No adjustment.
### Epoch 40 - freeze
- Reference metrics: `{'precision': 0.64491, 'recall': 0.63083, 'map50': 0.6639, 'map50_95': 0.42895}`
- Delta metrics: `{'precision': 0.04469000000000001, 'recall': 0.058660000000000045, 'map50': 0.05858999999999992, 'map50_95': 0.06047999999999998}`
- Last safe policy id: `initial`
- Rollback reason: `None`
- Group budget before: `{'photometric': {'used': 0.16, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.03, 'budget': 0.08, 'within_budget': True}}`
- Group budget after: `{'photometric': {'used': 0.16, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.03, 'budget': 0.08, 'within_budget': True}}`
- Trust-region clipping: `[]`
- No adjustment.
### Epoch 45 - freeze
- Reference metrics: `{'precision': 0.66928, 'recall': 0.72503, 'map50': 0.69559, 'map50_95': 0.44787}`
- Delta metrics: `{'precision': 0.07050000000000001, 'recall': -0.01800999999999997, 'map50': 0.058620000000000005, 'map50_95': 0.07294999999999996}`
- Last safe policy id: `initial`
- Rollback reason: `None`
- Group budget before: `{'photometric': {'used': 0.16, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.03, 'budget': 0.08, 'within_budget': True}}`
- Group budget after: `{'photometric': {'used': 0.16, 'budget': 0.3, 'within_budget': True}, 'texture': {'used': 0.14, 'budget': 0.35, 'within_budget': True}, 'occlusion': {'used': 0.03, 'budget': 0.08, 'within_budget': True}}`
- Trust-region clipping: `[]`
- No adjustment.
