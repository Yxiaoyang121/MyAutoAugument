# In-Loop YOLO Default Feedback Report

## Run Integrity

- Training success: `true`
- Single-run continuous training: `true`
- Stage restart count: `0`
- Epoch sequence continuous: `true`
- Epoch sequence: `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`
- YOLO default augmentation enabled: `true`
- Industrial augmentation enabled: `true`
- Industrial augmentation dynamic: `false`
- Train image count: `2071`
- Fixed augmented dataset generated: `false`
- BBox/class legal: `true`
- Global best.pt: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\cp_catf_paper_mode_10ep_smoke\train\weights\best.pt`

## Metrics

- Precision: `0.7728`
- Recall: `0.4190`
- mAP50: `0.5140`
- mAP50-95: `0.3242`

## Clean Native Reference

- Baseline: `clean_native_yolo_default`
- Precision: `0.7262`
- Recall: `0.6844`
- mAP50: `0.7616`
- mAP50-95: `0.5250`

## Feedback

- Feedback enabled: `true`
- Feedback controller: `CATF-v2`
- CATF-v2 safe mode enabled: `false`
- CATF-v2 gated mode enabled: `false`
- CATF-v2 rollback mode enabled: `false`
- CATF-v2 RiskGuard enabled: `false`
- CATF-v2 causal probe enabled: `true`
- Offline causal probe decisions used: `false`
- RiskGuard blocked ops: `0`
- RiskGuard sampler-only fallback: `false`
- Adaptive burn-in enabled: `false`
- Adaptive start epoch: `None`
- Adaptive candidate started: `false`
- Adaptive no-op fallback: `false`
- RB rollback triggered: `false`
- Safe no-op fallback triggered: `false`
- Safe controller reasons: `none`
- Gated no-op fallback triggered: `false`
- Gated controller reasons: `none`
- Reference curve loaded: `true`
- Reference curve path: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\clean_native_yolo_default_seed42_50ep\train\results.csv`
- Feedback epochs: `[5]`
- Policy updates: `1`
- copy_paste status: `pending_object_bank_design`
- Ops upregulated: `local_contrast:4, sharpen_mild:4`
- Ops downregulated: `none`
- Guard-triggered epochs: `5:causal_probe_no_candidate_passed`
- Rollback triggered: `false`
- Cooldown triggered: `false`
- Freeze triggered: `false`

## Industrial Augmentation Stats

| op | seen | applied | skipped_probability |
|---|---:|---:|---:|

## Constraint Scoring

- Baseline: `clean_native_yolo_default`
- Delta Precision: `0.0466`
- Delta Recall: `-0.2654`
- Delta mAP50: `-0.2476`
- Delta mAP50-95: `-0.2008`
- constraint_failed: `true`

## Verdict

- Exceeds clean native reference on Precision: `true`
- Exceeds clean native reference on Recall: `false`
- Exceeds clean native reference on mAP50: `false`
- mAP50-95 remains within constraint: `false`
Not acceptable as the paper main method under current industrial constraints.
