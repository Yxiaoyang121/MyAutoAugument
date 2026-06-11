# Seed0 CP-CATF Paper-Mode Execution Flow Report

## Summary

- Run root: `outputs\experiments\cp_catf_paper_mode_execution_fixed_seed0_only`
- Scope: seed0 CP-CATF paper-mode only; no clean rerun, no seed1/seed2, no multiseed summary.
- `candidate_policy_1_roi_texture` accept seen: `True` at epochs `[25]`.
- Executable policy generated after accept: `True`.
- Sample router called after accept: `True`.
- Industrial image samples augmented: `226`.
- ROI applied: `312`.
- Router random draw count: `1330`.
- Final val leakage: `False`.
- BBox/class legality: invalid bbox `0`, bbox OOB `0`, class OOB `0`.

## Metrics

| metric | clean paper seed0 | CP-CATF seed0 | delta |
|---|---:|---:|---:|
| P | 0.751343 | 0.739931 | -0.011412 |
| R | 0.676301 | 0.676311 | +0.000010 |
| mAP50 | 0.756646 | 0.759345 | +0.002699 |
| mAP50-95 | 0.511423 | 0.502791 | -0.008631 |

- `constraint_failed=True`; reasons: `['precision_drop_gt_0.01']`.

## Per-Epoch Flow

| epoch | decision | action | active class | ops | causal score | executable policy | router called | router candidates | router selected | ROI applied | industrial applied | random draws | status | no-op reason | leakage false |
|---:|---|---|---:|---|---:|---|---|---|---|---:|---:|---:|---|---|---|
| 5 | `candidate_policy_3_sampler_only` | `sampler_only` | 8 | `` | -0.020000 | `False` | `False` | `0` | `0` | 0 | 0 | 0 | `sampler_only_or_rejected_strict_image_noop` | offline_causal_probe_sampler_only_image_noop | `True` |
| 10 | `candidate_policy_3_sampler_only` | `sampler_only` | 8 | `` | 0.007500 | `False` | `False` | `0` | `0` | 0 | 0 | 0 | `sampler_only_or_rejected_strict_image_noop` | offline_causal_probe_sampler_only_image_noop | `True` |
| 15 | `candidate_policy_3_sampler_only` | `sampler_only` | 12 | `` | 0.007500 | `False` | `False` | `0` | `0` | 0 | 0 | 0 | `sampler_only_or_rejected_strict_image_noop` | offline_causal_probe_sampler_only_image_noop | `True` |
| 20 | `candidate_policy_3_sampler_only` | `sampler_only` | 9 | `` | 0.011667 | `False` | `False` | `0` | `0` | 0 | 0 | 0 | `sampler_only_or_rejected_strict_image_noop` | offline_causal_probe_sampler_only_image_noop | `True` |
| 25 | `candidate_policy_1_roi_texture` | `accept` | 9 | `sharpen_mild,local_contrast` | 0.017263 | `True` | `True` | `665` | `226` | 312 | 226 | 1330 | `accepted_image_aug_executed` |  | `True` |
| 30 | `candidate_policy_3_sampler_only` | `sampler_only` | None | `` | 0.000000 | `False` | `False` | `0` | `0` | 0 | 0 | 0 | `sampler_only_or_rejected_strict_image_noop` | offline_causal_probe_sampler_only_image_noop | `True` |
| 35 | `candidate_policy_3_sampler_only` | `sampler_only` | None | `` | 0.000000 | `False` | `False` | `0` | `0` | 0 | 0 | 0 | `sampler_only_or_rejected_strict_image_noop` | offline_causal_probe_sampler_only_image_noop | `True` |
| 40 | `candidate_policy_3_sampler_only` | `sampler_only` | None | `` | 0.000000 | `False` | `False` | `0` | `0` | 0 | 0 | 0 | `sampler_only_or_rejected_strict_image_noop` | offline_causal_probe_sampler_only_image_noop | `True` |
| 45 | `candidate_policy_3_sampler_only` | `sampler_only` | None | `` | 0.000000 | `False` | `False` | `0` | `0` | 0 | 0 | 0 | `sampler_only_or_rejected_strict_image_noop` | offline_causal_probe_sampler_only_image_noop | `True` |

## Execution Evidence

- Epoch 25 accepted `candidate_policy_1_roi_texture` for class `9` with `sharpen_mild` and `local_contrast`.
- `policy_matrix_epoch_25_after_causal_probe.json` contains executable class 9 ops: `sharpen_mild(prob=0.20,strength=0.22)` and `local_contrast(prob=0.18,strength=0.20)`.
- `policy_history.json` for epoch 25 records `active_classes=[9]` and no `causal_probe_no_candidate_passed` guard.
- `online_aug_stats.json` records op execution: `local_contrast` applied `115` times and `sharpen_mild` applied `141` times.
- `roi_aug_stats.json` records `312` ROI applications, all on class `9`.
- Per-epoch router candidate/selected sample counts are not currently logged. The report uses global execution-window evidence: per-op `seen=665`, `samples_augmented=226`, and `router_random_draw_count=1330`.

## Conclusion

The paper-mode CP-CATF accept-to-execution chain is now functionally connected for seed0: causal probe accept produced an executable policy, the sample router drew candidates, ROI augmentation ran, and online stats recorded actual image augmentation. This seed0 execution validation is not a multiseed result. Its final metric profile fails the industrial constraint on Precision only, so seed1/seed2 should not be launched as a performance validation without first deciding whether this precision risk is acceptable for the next diagnostic step.
