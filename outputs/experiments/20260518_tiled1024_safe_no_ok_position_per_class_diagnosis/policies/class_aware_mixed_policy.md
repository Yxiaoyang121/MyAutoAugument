# Class-Aware Mixed Policy

- Policy ID: `class_aware_policy_001`
- Type: `class_aware_mixed_policy`
- Scope: policy generation only; no augmented dataset was built and no training was run.

## Branches

### photometric_branch

- Target classes: `加强筋打伤, 开裂, 油污, 浅划伤, 漏背锡, 碰伤, 脏污, 轮廓划伤, 锡尖`
- Reason: low_contrast_error / dark_object_error
- Avg severity/confidence/priority: `0.433` / `0.683` / `0.274`

| operation | prob | strength | params |
|---|---:|---:|---|
| `clahe` | 0.352 | 0.500 | `{"clip_limit_range": [2.0, 3.0], "tile_grid_size": 8, "target_class_ids": [2, 3, 4, 5, 6, 7, 8, 9, 11]}` |
| `contrast` | 0.352 | 0.500 | `{"alpha_range": [1.1, 1.3], "target_class_ids": [2, 3, 4, 5, 6, 7, 8, 9, 11]}` |
| `gamma` | 0.352 | 0.500 | `{"gamma_range": [0.75, 0.9], "target_class_ids": [2, 3, 4, 5, 6, 7, 8, 9, 11]}` |
| `brightness` | 0.352 | 0.500 | `{"beta_range": [10, 30], "target_class_ids": [2, 3, 4, 5, 6, 7, 8, 9, 11]}` |
| `sharpen_mild` | 0.352 | 0.500 | `{"amount_range": [0.5, 0.9], "sigma": 1.0, "target_class_ids": [2, 3, 4, 5, 6, 7, 8, 9, 11]}` |

Target evidence:

- `加强筋打伤`: primary=`low_support_class`, priority=`0.099`, attributions=`low_support_class, class_sample_imbalance, low_contrast_error, dark_object_error, small_object_error`
- `开裂`: primary=`low_support_class`, priority=`0.164`, attributions=`low_support_class, class_sample_imbalance, dark_object_error, class_confusion`
- `油污`: primary=`high_fp_class`, priority=`0.379`, attributions=`low_recall_class, high_fp_class, low_contrast_error, dark_object_error, texture_confusion, tile_edge_error, class_confusion`
- `浅划伤`: primary=`class_sample_imbalance`, priority=`0.251`, attributions=`class_sample_imbalance, low_recall_class, low_contrast_error, texture_confusion, small_object_error, class_confusion`
- `漏背锡`: primary=`low_contrast_error`, priority=`0.166`, attributions=`low_contrast_error, weak_localization, class_confusion`
- `碰伤`: primary=`low_recall_class`, priority=`0.439`, attributions=`low_recall_class, low_contrast_error, dark_object_error, texture_confusion, weak_localization, small_object_error, tile_edge_error, class_confusion`
- `脏污`: primary=`high_fp_class`, priority=`0.375`, attributions=`low_recall_class, high_fp_class, low_contrast_error, dark_object_error, texture_confusion, weak_localization, small_object_error, class_confusion`
- `轮廓划伤`: primary=`low_recall_class`, priority=`0.441`, attributions=`low_recall_class, low_contrast_error, dark_object_error, texture_confusion, weak_localization, small_object_error, tile_edge_error, class_confusion`
- `锡尖`: primary=`low_contrast_error`, priority=`0.150`, attributions=`low_contrast_error, small_object_error, class_confusion`

### copy_paste_branch

- Target classes: `加强筋打伤, 开裂, 浅划伤, 锡丝残留`
- Reason: low_support_class / class_sample_imbalance
- Avg severity/confidence/priority: `0.497` / `0.438` / `0.198`

| operation | prob | strength | params |
|---|---:|---:|---|
| `class_balanced_copy_paste` | 0.313 | 0.608 | `{"target_class_ids": [2, 3, 5, 10], "prefer_small": true, "class_balanced": true, "max_paste_count": 2, "max_overlap": 0.2}` |

Target evidence:

- `加强筋打伤`: primary=`low_support_class`, priority=`0.099`, attributions=`low_support_class, class_sample_imbalance, low_contrast_error, dark_object_error, small_object_error`
- `开裂`: primary=`low_support_class`, priority=`0.164`, attributions=`low_support_class, class_sample_imbalance, dark_object_error, class_confusion`
- `浅划伤`: primary=`class_sample_imbalance`, priority=`0.251`, attributions=`class_sample_imbalance, low_recall_class, low_contrast_error, texture_confusion, small_object_error, class_confusion`
- `锡丝残留`: primary=`class_sample_imbalance`, priority=`0.278`, attributions=`class_sample_imbalance, low_recall_class, small_object_error, tile_edge_error, class_confusion`

### texture_branch

- Target classes: `油污, 浅划伤, 碰伤, 脏污, 轮廓划伤, 锡膏`
- Reason: texture_confusion / weak texture boundary
- Avg severity/confidence/priority: `0.446` / `0.800` / `0.351`

| operation | prob | strength | params |
|---|---:|---:|---|
| `sharpen` | 0.398 | 0.515 | `{"amount_range": [0.5, 0.9], "sigma": 1.0, "target_class_ids": [4, 5, 7, 8, 9, 12]}` |
| `local_contrast` | 0.398 | 0.515 | `{"clip_limit_range": [1.5, 2.5], "target_class_ids": [4, 5, 7, 8, 9, 12]}` |
| `mild_noise` | 0.358 | 0.445 | `{"std_range": [0.01, 0.03], "target_class_ids": [4, 5, 7, 8, 9, 12]}` |

Target evidence:

- `油污`: primary=`high_fp_class`, priority=`0.379`, attributions=`low_recall_class, high_fp_class, low_contrast_error, dark_object_error, texture_confusion, tile_edge_error, class_confusion`
- `浅划伤`: primary=`class_sample_imbalance`, priority=`0.251`, attributions=`class_sample_imbalance, low_recall_class, low_contrast_error, texture_confusion, small_object_error, class_confusion`
- `碰伤`: primary=`low_recall_class`, priority=`0.439`, attributions=`low_recall_class, low_contrast_error, dark_object_error, texture_confusion, weak_localization, small_object_error, tile_edge_error, class_confusion`
- `脏污`: primary=`high_fp_class`, priority=`0.375`, attributions=`low_recall_class, high_fp_class, low_contrast_error, dark_object_error, texture_confusion, weak_localization, small_object_error, class_confusion`
- `轮廓划伤`: primary=`low_recall_class`, priority=`0.441`, attributions=`low_recall_class, low_contrast_error, dark_object_error, texture_confusion, weak_localization, small_object_error, tile_edge_error, class_confusion`
- `锡膏`: primary=`low_recall_class`, priority=`0.221`, attributions=`low_recall_class, texture_confusion, weak_localization, small_object_error, class_confusion`

### localization_branch

- Target classes: `漏背锡, 碰伤, 脏污, 轮廓划伤, 锡膏`
- Reason: weak_localization
- Avg severity/confidence/priority: `0.367` / `0.880` / `0.329`

| operation | prob | strength | params |
|---|---:|---:|---|
| `mild_translate` | 0.356 | 0.435 | `{"max_translate": 0.04, "target_class_ids": [6, 7, 8, 9, 12]}` |
| `mild_scale` | 0.376 | 0.485 | `{"scale_range": [0.92, 1.08], "target_class_ids": [6, 7, 8, 9, 12]}` |

Target evidence:

- `漏背锡`: primary=`low_contrast_error`, priority=`0.166`, attributions=`low_contrast_error, weak_localization, class_confusion`
- `碰伤`: primary=`low_recall_class`, priority=`0.439`, attributions=`low_recall_class, low_contrast_error, dark_object_error, texture_confusion, weak_localization, small_object_error, tile_edge_error, class_confusion`
- `脏污`: primary=`high_fp_class`, priority=`0.375`, attributions=`low_recall_class, high_fp_class, low_contrast_error, dark_object_error, texture_confusion, weak_localization, small_object_error, class_confusion`
- `轮廓划伤`: primary=`low_recall_class`, priority=`0.441`, attributions=`low_recall_class, low_contrast_error, dark_object_error, texture_confusion, weak_localization, small_object_error, tile_edge_error, class_confusion`
- `锡膏`: primary=`low_recall_class`, priority=`0.221`, attributions=`low_recall_class, texture_confusion, weak_localization, small_object_error, class_confusion`
