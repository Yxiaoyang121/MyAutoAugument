# In-Loop YOLO Default Feedback Report

## Run Integrity

- Training success: `true`
- Single-run continuous training: `true`
- Stage restart count: `0`
- Epoch sequence continuous: `true`
- Epoch sequence: `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`
- YOLO default augmentation enabled: `true`
- Industrial augmentation enabled: `true`
- Industrial augmentation dynamic: `true`
- Train image count: `2301`
- Fixed augmented dataset generated: `false`
- BBox/class legal: `true`
- Global best.pt: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\catf_v2_activation_fixed_10ep_smoke\train\weights\best.pt`

## Metrics

- Precision: `0.6324`
- Recall: `0.4960`
- mAP50: `0.5659`
- mAP50-95: `0.3750`

## Clean Native Reference

- Baseline: `clean_native_yolo_default`
- Precision: `0.7262`
- Recall: `0.6844`
- mAP50: `0.7616`
- mAP50-95: `0.5250`

## Feedback

- Feedback enabled: `true`
- Feedback controller: `CATF-v2`
- Reference curve loaded: `true`
- Reference curve path: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\clean_native_yolo_default_seed42_50ep\train\results.csv`
- Feedback epochs: `[5]`
- Policy updates: `1`
- copy_paste status: `pending_object_bank_design`
- Ops upregulated: `local_contrast:4, sharpen_mild:4`
- Ops downregulated: `none`
- Guard-triggered epochs: `none`
- Rollback triggered: `false`
- Cooldown triggered: `false`
- Freeze triggered: `false`

## Industrial Augmentation Stats

| op | seen | applied | skipped_probability |
|---|---:|---:|---:|
| local_contrast | 1215 | 15 | 1200 |
| sharpen_mild | 1215 | 5 | 1209 |

## Constraint Scoring

- Baseline: `clean_native_yolo_default`
- Delta Precision: `-0.0938`
- Delta Recall: `-0.1884`
- Delta mAP50: `-0.1957`
- Delta mAP50-95: `-0.1501`
- constraint_failed: `true`

## Verdict

- Exceeds clean native reference on Precision: `false`
- Exceeds clean native reference on Recall: `false`
- Exceeds clean native reference on mAP50: `false`
- mAP50-95 remains within constraint: `false`
Not acceptable as the paper main method under current industrial constraints.
