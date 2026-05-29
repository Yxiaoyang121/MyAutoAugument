# CATF In-Loop Feedback Policy History

| epoch | action | guards | adjustments | frozen | copy_paste_status |
|---:|---|---|---:|---|---|
| 5 | accept | precision_guard,low_contrast_fn | 10 | false | pending_object_bank_design |

## Adjustments

### Epoch 5 - accept
- Reference metrics: `{'precision': 0.62773, 'recall': 0.36765, 'map50': 0.3489, 'map50_95': 0.21083}`
- Delta metrics: `{'precision': 0.06957000000000002, 'recall': 0.0022600000000000398, 'map50': 0.06698000000000004, 'map50_95': 0.046520000000000034}`
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
