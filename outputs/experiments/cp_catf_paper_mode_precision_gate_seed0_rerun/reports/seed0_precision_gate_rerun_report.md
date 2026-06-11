# Seed0 CP-CATF Precision-Aware Gate Rerun

## Dry Run Result

- Epoch 25 roi_texture rejected: `true`.
- estimated_precision_drop: `0.0300`.
- non_active_fp_delta: `0.0500`.
- high_confidence_fp_delta: `0.0500`.
- Selected after replay: `candidate_policy_3_sampler_only`.

## Rerun Candidate Decisions

| epoch | selected candidate | action | class | score | image aug | sampler effective | reason |
|---:|---|---|---:|---:|---|---|---|
| 5 | `candidate_policy_3_sampler_only` | `sampler_only` | 8 | -0.1520 | `false` | `false` | `offline_causal_probe_sampler_only_image_noop` |
| 10 | `candidate_policy_3_sampler_only` | `sampler_only` | 8 | -0.1005 | `false` | `false` | `offline_causal_probe_sampler_only_image_noop` |
| 15 | `candidate_policy_3_sampler_only` | `sampler_only` | 12 | -0.1245 | `false` | `false` | `offline_causal_probe_sampler_only_image_noop` |
| 20 | `candidate_policy_3_sampler_only` | `sampler_only` | 9 | -0.0963 | `false` | `false` | `offline_causal_probe_sampler_only_image_noop` |
| 25 | `candidate_policy_3_sampler_only` | `sampler_only` | 9 | -0.1172 | `false` | `false` | `offline_causal_probe_sampler_only_image_noop` |
| 30 | `candidate_policy_3_sampler_only` | `sampler_only` | -1 | 0.0000 | `false` | `false` | `offline_causal_probe_sampler_only_image_noop` |
| 35 | `candidate_policy_3_sampler_only` | `sampler_only` | -1 | 0.0000 | `false` | `false` | `offline_causal_probe_sampler_only_image_noop` |
| 40 | `candidate_policy_3_sampler_only` | `sampler_only` | 6 | -0.0857 | `false` | `false` | `offline_causal_probe_sampler_only_image_noop` |
| 45 | `candidate_policy_3_sampler_only` | `sampler_only` | -1 | 0.0000 | `false` | `false` | `offline_causal_probe_sampler_only_image_noop` |

## Execution Stats

- ROI applied: `0`.
- Industrial image augmented: `0`.
- Router random draw count: `0`.
- BBox/class legal: `true`.

## Final Val Leakage

- Final val leakage: `false`.
- Final val used for policy selection: `false`.
- train_core/probe overlap: `0`.
- probe/final val overlap: `0`.
- train_core/final val overlap: `0`.

## Metrics Against Paper Clean Seed0

| metric | clean paper seed0 | CP-CATF precision gate seed0 | delta |
|---|---:|---:|---:|
| P | 0.7513 | 0.7513 | 0.0000 |
| R | 0.6763 | 0.6763 | 0.0000 |
| mAP50 | 0.7566 | 0.7566 | 0.0000 |
| mAP50-95 | 0.5114 | 0.5114 | 0.0000 |

- constraint_failed vs paper clean seed0: `false`.
- failure reasons vs paper clean seed0: `none`.

## Interpretation

- The precision-aware gate blocks the known epoch 25 roi_texture risk before training execution.
- The rerun does not execute image augmentation: all feedback points select sampler_only, and sampler weighting is still pending dataloader support, so the image path is strict no-op.
- The trainer's legacy constraint artifact still compares against an older clean_native_yolo_default reference; this report uses the requested paper clean seed0 baseline.
- Recommendation to continue seed1/seed2: `false`.
- Recommendation reason: seed0 passes the requested paper-clean constraint, but the rerun is strict image no-op because all image candidates were rejected and sampler-only is pending; do not proceed to formal seed1/seed2 enhancement validation until an accepted image candidate can pass the precision gate or sampler-only is actually wired into the dataloader.
