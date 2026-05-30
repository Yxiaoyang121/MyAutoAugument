# CATF-v2 Activation Fixed Smoke Report

Generated: 2026-05-31T00:40:43

## Run

- Output: `outputs/experiments/catf_v2_activation_fixed_10ep_smoke/`
- Train success: `true`
- Val success: `true`
- Epoch continuous: `true`
- Feedback epochs: `[5]`
- Train images: `2301`
- Fixed augmented dataset generated: `false`
- YOLO default augmentation enabled: `true`
- top_k_active_classes: `2`

## Activation Checks

1. OK2/OK3 default no_aug/stable:
   - OK2 status=`guarded`, no_aug=`True`, domain_high_fp_prior=`True`, high_fp_guarded=`True`.
   - OK3 status=`guarded`, no_aug=`True`, domain_high_fp_prior=`True`, high_fp_guarded=`True`.

2. OK3 active state:
   - OK3 active: `false`
   - OK3 ops with nonzero prob: `{}`
   - OK3 ROI applied: `0`
   - Decision: OK3 is blocked from active top-k and ROI enhancement.

3. 漏背锡 activation:
   - status=`active`, dominant_issue=`texture_boundary_weak`
   - diagnosis: Recall=`0.2826086956521739`, FN=`32`, evidence_count=`39`, diagnosis_confidence=`0.96`, low_contrast_fn_count=`16`
   - active ops: `{'sharpen_mild': 0.015, 'local_contrast': 0.015}`
   - Decision: 漏背锡 remains reasonably activated with conservative ROI sharpen/local_contrast.

4. 脏污 domain high-FP prior:
   - status=`active`, domain_high_fp_prior=`True`, dominant_issue=`low_recall`
   - photometric probs clahe/gamma/brightness/contrast: `{'clahe': 0.0, 'gamma': 0.0, 'brightness': 0.0, 'contrast': 0.0}`
   - threshold recommendation: `0.25`, reason=`domain_high_fp_prior_keep_threshold`
   - Decision: 脏污 is treated as domain high-FP prior; strong photometric and threshold lowering are avoided.

5. Active classes:
   - `[('6', '漏背锡', 'texture_boundary_weak'), ('8', '脏污', 'low_recall')]`

6. ROI augmentation:
   - roi_aug_applied=`23`
   - roi_aug_skipped_small_roi=`0`
   - roi_aug_skipped_conflict=`2`
   - affected_classes=`{'6': 20, '8': 3}`
   - OK3 ROI applied=`0`

7. BBox/class legality:
   - invalid_bbox_count=`0`
   - class_id_oob_count=`0`
   - bbox_class_valid=`true`

## Conclusion

The activation rules now block OK2/OK3 from active top-k and ROI augmentation, keep low-support classes in observe/pending status, and prioritize the non-prior defect class 漏背锡 over domain high-FP prior classes. 脏污 remains domain-prior constrained: only very conservative texture ROI ops are allowed and threshold lowering is blocked. This smoke is suitable to proceed to a CATF-v2 seed42 50 epoch validation, while treating the 10 epoch metric constraint flag as non-decisive because it compares a short smoke run with the 50 epoch clean reference.

## Key Files

- policy_history: `outputs/experiments/catf_v2_activation_fixed_10ep_smoke/reports/policy_history.json`
- policy_matrix_after: `outputs/experiments/catf_v2_activation_fixed_10ep_smoke/reports/policy_matrix_epoch_5_after.json`
- ROI stats: `outputs/experiments/catf_v2_activation_fixed_10ep_smoke/reports/roi_aug_stats.json`
- threshold calibration: `outputs/experiments/catf_v2_activation_fixed_10ep_smoke/reports/threshold_calibration.json`
