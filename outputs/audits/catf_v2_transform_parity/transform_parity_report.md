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
- clean vs CATF-v2 formal-force-skip final output identical: `false`
- raw vs CATF-v2 noop intermediate identical: `true`
- raw vs CATF-v2 formal-force-skip intermediate identical: `false`
- clean vs noop final mismatches: `0`
- clean vs formal-force-skip final mismatches: `1`
- raw vs formal-force-skip intermediate mismatches: `89`

## Rewrite And Random Audit

- CATF-v2 noop instance rewrites: `0`
- CATF-v2 formal-force-skip instance rewrites: `100`
- CATF-v2 formal-force-skip image rewrites: `100`
- CATF-v2 formal-force-skip cls rewrites: `100`
- CATF-v2 formal-force-skip router random draws: `0`
- CATF-v2 formal-force-skip random state changed samples: `0`
- CATF-v2 formal-force-skip applied ops: `0`
- CATF-v2 formal-force-skip samples seen by router: `100`
- CATF-v2 formal-force-skip bbox_oob_count: `1`
- CATF-v2 formal-force-skip invalid_bbox_count: `0`
- CATF-v2 formal-force-skip class_id_oob_count: `0`

## Answers

1. clean vs CATF-v2 noop is fully identical: `true`.
2. clean vs CATF-v2 formal-force-skip final output is fully identical: `false`.
3. Formal-force-skip does replace image/cls/Instances objects before the native YOLO transform, because the online transform converts boxes through the router and rebuilds `Instances` even when no op is applied.
4. That rewrite did not change final image hash, class order, bbox hash, dtype, shape, or sample order in the sampled paths: `false`.
5. Router random draws under formal-force-skip: `0`.
6. no_aug/stable/high-FP samples can enter router validation in formal-force-skip, but no industrial or ROI operation is applied and no probability draw is consumed.
7. Router validation clipped or flagged out-of-bound boxes in `1` sampled case(s), which is enough to change a final YOLO training sample even with zero applied augmentation.

## Interpretation

- Root cause classification: `formal_force_skip_changes_final_training_sample`.
- The final training sample differs under force-skip; CATF-v2 should bypass the online transform unless an op is actually selected.

## Fix Suggestions

- Keep `catf_noop=true` bypass as the strict parity path.
- For formal CATF-v2, add an early bypass when sample routing finds no active op before converting/rebuilding `Instances`.
- Only copy image, cls, and `Instances` after a specific ROI/industrial op is selected for application.
- Force-skip paths should continue to avoid random draws and should not call validation that can clip/filter boxes.

## Artifacts

- JSON summary: `outputs/audits/catf_v2_transform_parity/transform_parity.json`
- Per-sample diffs: `outputs/audits/catf_v2_transform_parity/diff_samples/`
