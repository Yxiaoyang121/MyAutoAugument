# CATF-v2 Transform-Level Parity Audit

## Scope

- Data: `E:\TJGY\MinPaper\MyAutoAugument\outputs\datasets\tiled\tiled_1024_ov20_full_safe_no_ok_position\data.yaml`
- Samples compared: `100`
- Paths: clean YOLO default, CATF-v2 noop, CATF-v2 formal-force-skip.
- No training was run.

## Sample Coverage

- stable_no_aug: `42`
- active_defect: `23`
- domain_high_fp_prior: `23`
- low_support: `10`
- multi_class: `38`
- empty_label: `10`

## Parity Results

- clean vs CATF-v2 noop final output identical: `true`
- clean vs CATF-v2 formal-force-skip final output identical: `true`
- raw vs CATF-v2 noop intermediate identical: `true`
- raw vs CATF-v2 formal-force-skip intermediate identical: `true`
- clean vs noop final mismatches: `0`
- clean vs formal-force-skip final mismatches: `0`
- raw vs formal-force-skip intermediate mismatches: `0`

## Rewrite And Random Audit

- CATF-v2 noop instance rewrites: `0`
- CATF-v2 formal-force-skip instance rewrites: `0`
- CATF-v2 formal-force-skip image rewrites: `0`
- CATF-v2 formal-force-skip cls rewrites: `0`
- CATF-v2 formal-force-skip router random draws: `0`
- CATF-v2 formal-force-skip random state changed samples: `0`
- CATF-v2 formal-force-skip applied ops: `0`
- CATF-v2 formal-force-skip samples seen by router: `100`
- CATF-v2 formal-force-skip bbox_oob_count: `0`
- CATF-v2 formal-force-skip invalid_bbox_count: `0`
- CATF-v2 formal-force-skip class_id_oob_count: `0`

## Answers

1. clean vs CATF-v2 noop is fully identical: `true`.
2. clean vs CATF-v2 formal-force-skip final output is fully identical: `true`.
3. Formal-force-skip does not replace image/cls/Instances objects before the native YOLO transform.
4. Final image hash, class order, bbox hash, dtype, shape, and sample order match clean native: `true`.
5. Router random draws under formal-force-skip: `0`.
6. no_aug/stable/high-FP force-skip samples do not apply industrial/ROI operations and do not consume probability draws.
7. Router validation clipped or flagged out-of-bound boxes in `0` sampled case(s).

## Interpretation

- Root cause classification: `no_transform_level_difference_detected`.
- This audit does not support transform-level label rewrite as the explanation for seed1 CATF-v2 ROI/industrial=0 but different final metrics.
- The formal-force-skip path now removes the previously identified mechanism risk: it does not rewrite labels or clip boxes when no augmentation is applied.

## Bypass Guardrails

- Keep `catf_noop=true` bypass as the strict parity path.
- Formal CATF-v2 should continue to bypass rewrite when sample routing finds no active op before converting/rebuilding `Instances`.
- Only copy image, cls, and `Instances` after a specific ROI/industrial op is selected for application.
- Force-skip paths should continue to avoid random draws and should not call validation that can clip/filter boxes.

## Artifacts

- JSON summary: `outputs/audits/catf_v2_transform_parity/transform_parity.json`
- Per-sample diffs: `outputs/audits/catf_v2_transform_parity/diff_samples/`
