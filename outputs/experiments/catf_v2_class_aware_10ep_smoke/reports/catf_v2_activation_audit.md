# CATF-v2 Activation Audit

## Scope

- Source: `outputs/experiments/catf_v2_class_aware_10ep_smoke/reports/`
- Training run started: `false`
- Purpose: audit epoch 5 CATF-v2 activation before any formal 50 epoch run.

## Activation Summary

- Active classes: `[1, 6, 8]`
- Frozen classes: `[]`
- High-FP guarded classes: `[]`
- ROI applied: `150`
- ROI affected classes: `{'1': 122, '6': 25, '8': 3}`

## Class 1 OK3

- Dominant issue: `low_contrast_fn`
- Evidence count: `53`
- Diagnosis confidence: `1.0`
- TP/FP/FN: `271/49/2`
- Precision/Recall: `0.8469/0.9891`
- Flags: high_fp=`false`, low_recall=`false`, low_contrast=`true`, weak_localization=`true`
- Activated ops: `{'gamma': {'prob': 0.01, 'strength': 0.015, 'max_prob': 0.15, 'max_strength': 0.45}, 'local_contrast': {'prob': 0.015, 'strength': 0.02, 'max_prob': 0.18, 'max_strength': 0.45}}`

Assessment: OK3 activation is not reasonable for a formal run. The trigger came from `low_contrast_fn`, but only 2 FN exist while FP is 49 and Recall is already 0.9891. Because OK-like classes are non-defect/stable references, activating ROI photometric/texture augmentation on OK3 risks amplifying background/normal texture and precision instability.

## Class 6 `漏背锡`

- Dominant issue: `texture_boundary_weak`; secondary: `['weak_localization', 'low_recall', 'low_contrast_fn']`
- low_contrast_fn: `true`; low_contrast_fn_count: `16`
- TP/FP/FN: `13/7/32`; Precision/Recall: `0.6500/0.2826`
- Evidence count/confidence: `39` / `0.96`
- Activated ops: `{'sharpen_mild': {'prob': 0.015, 'strength': 0.02, 'max_prob': 0.2, 'max_strength': 0.45}, 'local_contrast': {'prob': 0.015, 'strength': 0.02, 'max_prob': 0.18, 'max_strength': 0.45}}`

Assessment: class 6 `漏背锡` activation is reasonable. It has low Recall, many FN, medium support, low-contrast evidence, and weak-localization/texture-boundary evidence. The selected ops are conservative ROI `sharpen_mild` and `local_contrast`, not aggressive brightness/contrast.

## Class 8 `脏污`

- Dominant issue: `low_recall`; secondary: `['low_contrast_fn']`
- high_fp: `false`; FP count/rate: `0` / `0.0000`
- low_contrast_fn: `true`; low_contrast_fn_count: `21`
- TP/FP/FN: `0/0/42`; Precision/Recall: `0.0000/0.0000`
- Activated ops: `{'sharpen_mild': {'prob': 0.01, 'strength': 0.015, 'max_prob': 0.2, 'max_strength': 0.45}, 'local_contrast': {'prob': 0.01, 'strength': 0.015, 'max_prob': 0.18, 'max_strength': 0.45}}`
- Threshold calibration recommendation: `0.15` reason=`low_recall_lower_threshold`

Assessment: class 8 `脏污` activation is only partially reasonable. In this epoch it is not high-FP by observed counts because there are no predictions for the class, so the controller treats it as low Recall. However dirty/oil/stain-like classes are FP-prone by domain, so CATF-v2 should add a domain high-FP prior or at least avoid lowering thresholds and avoid photometric escalation until true FP behavior is known.

## Low-Support Classes 2 and 3

- Class 2 `加强筋打伤`: support=`low`, confidence=`0.28`, strong_update_allowed=`false`, ops=`{}`, copy_paste_candidate=`True`.
- Class 3 `开裂`: support=`low`, confidence=`0.07`, strong_update_allowed=`false`, ops=`{}`, copy_paste_candidate=`True`.

Assessment: class 2 and 3 did not trigger strong photometric augmentation. They are correctly treated as low-support observations with oversampling/copy-paste pending, though the policy matrix after file does not yet surface those pending flags because they were not among active top-k classes.

## Stable / No-Aug Rule Audit

| class_id | class_name | OK-like | high_perf | currently_stable | P | R | FP | FN |
|---:|---|---|---|---|---:|---:|---:|---:|
| 0 | OK2 | true | true | false | 0.8312 | 1.0000 | 13 | 0 |
| 1 | OK3 | true | true | false | 0.8469 | 0.9891 | 49 | 2 |

Assessment: OK2/OK3 should be treated as default stable/no_aug classes unless there is very strong class-specific evidence. Current stable logic is too strict and did not freeze them.

## ROI Augmentation Audit

- ROI applied: `150`
- Affected classes: `{'1': 122, '6': 25, '8': 3}`
- Acts on OK3: `true`
- Skipped small ROI: `0`
- Skipped conflict: `0`
- ROI implementation applies augmentation inside target bbox expanded by 1.5x and blends it back; bbox/class ids are unchanged.

## Recommendations

- Add no_aug_class or stable_by_name rule for OK2/OK3 unless class-specific FN evidence is very strong and FP is low.
- Do not allow low_contrast_fn dominance when FN count is tiny and FP count is much larger; require min low_contrast_fn_count >= 5 or FN_rate above threshold.
- Require diagnosis_confidence >= 0.5 plus strong_update_allowed plus no precision/FP warning before activation.
- For stain/dirty/oil-like classes, prefer high-FP/domain guard first; avoid threshold lowering and avoid global/full-image photometric.
- Lower top_k_active_classes from 3 to 2 for the next smoke to reduce simultaneous intervention blast radius.
- Keep class 6 activation but prefer ROI sharpen/local_contrast over gamma/brightness/contrast.

## Go / No-Go

- Recommendation before CATF-v2 seed42 50 epoch: `no-go until activation rules are tightened`.
- Main blocker: OK3 activation is not reasonable and stain-like class threshold/guard logic is too permissive.
