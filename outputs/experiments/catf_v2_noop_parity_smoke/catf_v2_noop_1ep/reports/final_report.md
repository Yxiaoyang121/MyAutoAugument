# In-Loop YOLO Default Feedback Report

## Run Integrity

- Training success: `true`
- Single-run continuous training: `true`
- Stage restart count: `0`
- Epoch sequence continuous: `true`
- Epoch sequence: `[1]`
- YOLO default augmentation enabled: `true`
- Industrial augmentation enabled: `true`
- Industrial augmentation dynamic: `false`
- Train image count: `2301`
- Fixed augmented dataset generated: `false`
- BBox/class legal: `true`
- Global best.pt: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\catf_v2_noop_parity_smoke\catf_v2_noop_1ep\train\weights\best.pt`

## Metrics

- Precision: `0.8277`
- Recall: `0.1977`
- mAP50: `0.2127`
- mAP50-95: `0.1464`

## Clean Native Reference

- Baseline: `clean_native_yolo_default`
- Precision: `0.7725`
- Recall: `0.6477`
- mAP50: `0.7542`
- mAP50-95: `0.4799`

## Feedback

- Feedback enabled: `true`
- Feedback controller: `CATF-v2`
- Reference curve loaded: `true`
- Reference curve path: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\multiseed_clean_yolo_default_vs_catf_v2\seed_1\clean_native_yolo_default\train\results.csv`
- Feedback epochs: `[]`
- Policy updates: `0`
- copy_paste status: `pending_object_bank_design`
- Ops upregulated: `none`
- Ops downregulated: `none`
- Guard-triggered epochs: `none`
- Rollback triggered: `false`
- Cooldown triggered: `false`
- Freeze triggered: `false`

## Industrial Augmentation Stats

| op | seen | applied | skipped_probability |
|---|---:|---:|---:|

## Constraint Scoring

- Baseline: `clean_native_yolo_default`
- Delta Precision: `0.0552`
- Delta Recall: `-0.4499`
- Delta mAP50: `-0.5415`
- Delta mAP50-95: `-0.3335`
- constraint_failed: `true`

## Verdict

- Exceeds clean native reference on Precision: `true`
- Exceeds clean native reference on Recall: `false`
- Exceeds clean native reference on mAP50: `false`
- mAP50-95 remains within constraint: `false`
Not acceptable as the paper main method under current industrial constraints.
