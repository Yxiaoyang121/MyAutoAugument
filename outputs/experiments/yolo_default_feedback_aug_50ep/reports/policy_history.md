# YOLO Default Feedback Policy History

| stage | adjustments | constraints |
|---:|---:|---|
| 0 | 25 | failed |

## Adjustments

### Stage 0
- `official_yolo_aug.hsv_v` value: 0.4000 -> 0.4150 (recall_low_or_fn_high)
- `official_yolo_aug.translate` value: 0.1000 -> 0.1050 (recall_low_or_fn_high)
- `official_yolo_aug.scale` value: 0.5000 -> 0.5250 (recall_low_or_fn_high)
- `custom_industrial_policy.clahe` prob: 0.0000 -> 0.0250 (recall_low_or_fn_high)
- `custom_industrial_policy.clahe` strength: 0.0000 -> 0.0200 (recall_low_or_fn_high)
- `custom_industrial_policy.gamma` prob: 0.0000 -> 0.0250 (recall_low_or_fn_high)
- `custom_industrial_policy.gamma` strength: 0.0000 -> 0.0200 (recall_low_or_fn_high)
- `custom_industrial_policy.sharpen_mild` prob: 0.0000 -> 0.0250 (recall_low_or_fn_high)
- `custom_industrial_policy.sharpen_mild` strength: 0.0000 -> 0.0200 (recall_low_or_fn_high)
- `official_yolo_aug.hsv_v` value: 0.4150 -> 0.3750 (precision_low_or_fp_high)
- `custom_industrial_policy.clahe` prob: 0.0250 -> 0.0000 (precision_low_or_fp_high)
- `custom_industrial_policy.clahe` strength: 0.0200 -> 0.0000 (precision_low_or_fp_high)
- `custom_industrial_policy.gamma` prob: 0.0250 -> 0.0000 (precision_low_or_fp_high)
- `custom_industrial_policy.gamma` strength: 0.0200 -> 0.0000 (precision_low_or_fp_high)
- `official_yolo_aug.erasing` value: 0.4000 -> 0.4400 (precision_low_hard_negative_like)
- `custom_industrial_policy.cutout_safe` prob: 0.0000 -> 0.0400 (precision_low_hard_negative_like)
- `custom_industrial_policy.cutout_safe` strength: 0.0000 -> 0.0300 (precision_low_hard_negative_like)
- `custom_industrial_policy.clahe` prob: 0.0000 -> 0.0250 (low_contrast_fn_high_monitor_fp)
- `custom_industrial_policy.clahe` strength: 0.0000 -> 0.0200 (low_contrast_fn_high_monitor_fp)
- `custom_industrial_policy.gamma` prob: 0.0000 -> 0.0250 (low_contrast_fn_high_monitor_fp)
- `custom_industrial_policy.gamma` strength: 0.0000 -> 0.0200 (low_contrast_fn_high_monitor_fp)
- `custom_industrial_policy.local_contrast` prob: 0.0000 -> 0.0250 (low_contrast_fn_high_monitor_fp)
- `custom_industrial_policy.local_contrast` strength: 0.0000 -> 0.0200 (low_contrast_fn_high_monitor_fp)
- `custom_industrial_policy.sharpen_mild` prob: 0.0250 -> 0.0500 (low_contrast_fn_high_monitor_fp)
- `custom_industrial_policy.sharpen_mild` strength: 0.0200 -> 0.0400 (low_contrast_fn_high_monitor_fp)
