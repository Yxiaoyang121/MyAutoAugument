# CP-CATF Paper-Mode Decision Coverage Audit

## Scope

- No training was run.
- No seed1/seed2 execution was launched.
- Replay used existing paper-mode diagnostic JSON from `outputs/experiments/multiseed_cp_catf_paper_mode`.
- Precision-aware decisions reuse current `build_probe_decision_from_rows` and `decide_candidate_acceptance` logic.
- `causal_score_accept_count` is the logged pre-precision image accept count from existing paper-mode events; `current_replay_causal_accept_count` is the stricter current-code replay count.
- Feedback epochs audited: `5, 10, 15, 20, 25, 30, 35, 40, 45` for seeds `0, 1, 2`.

## Leakage Status

- train_core images: `2071`.
- probe images: `230`.
- final val images: `677`.
- final val leakage detected: `false`.
- final val used for policy selection: `false`.

## Coverage Summary

| metric | value |
|---|---:|
| total_candidates | 81 |
| total_image_candidates | 54 |
| feedback_epoch_count | 27 |
| causal_score_accept_count | 8 |
| current_replay_causal_accept_count | 0 |
| precision_gate_reject_count | 8 |
| final_image_accept_count | 0 |
| final_sampler_only_count | 27 |
| final_effective_sampler_only_count | 0 |
| final_noop_count | 27 |
| image_accept_rate | 0.0000 |
| sampler_only_rate | 1.0000 |
| effective_sampler_only_rate | 0.0000 |
| noop_rate | 1.0000 |

## Seed Coverage

| seed | total candidates | image candidates | causal accept | precision reject | final image accept | sampler selected | effective sampler | strict no-op |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 27 | 18 | 1 | 1 | 0 | 9 | 0 | 9 |
| 1 | 27 | 18 | 2 | 2 | 0 | 9 | 0 | 9 |
| 2 | 27 | 18 | 5 | 5 | 0 | 9 | 0 | 9 |

## Rejection Reasons

| reason | count |
|---|---:|
| `estimated_precision_drop_too_high` | 20 |
| `non_active_fp_delta_too_high` | 20 |
| `high_confidence_fp_delta_too_high` | 20 |
| `fp_increase_rate_too_high` | 0 |
| `high_fp_spillover_rate_too_high` | 40 |
| `non_active_regression_too_high` | 40 |
| `ok_class_false_activation` | 0 |
| `bbox_instability_too_high` | 5 |
| `no_positive_benefit` | 54 |
| `insufficient_evidence` | 14 |

## Gate Conservatism Analysis

- All image candidates rejected under current precision-aware gate: `true`.
- Main requested rejection bucket: `no_positive_benefit` with `54` hits.
- Precision-gate rejects after original causal accept: `8`.
- Note: current replay rejects those legacy image accepts for the full risk stack; precision thresholds are present but not sole blockers.
- Precision reject reasons among those candidates: `{"estimated_precision_drop_too_high": 8, "high_confidence_fp_delta_too_high": 8, "non_active_fp_delta_too_high": 8}`.
- Sole precision blockers: `{}`.
- Candidates accepted if only high_confidence_fp_delta is relaxed: `0`.
- Candidates accepted if only non_active_fp_delta is relaxed: `0`.
- Candidates accepted if only estimated_precision_drop is relaxed: `0`.
- Candidates accepted if all three precision gates are relaxed: `0`.
- Attenuation candidates: `8`.

## High-Score Rejections

| seed | epoch | candidate | class | issue | logged/original score | current replay score | est P drop | non-active FP | high-conf FP | attenuation | reasons |
|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---|---|
| 0 | 25 | `candidate_policy_1_roi_texture` | 9 | `texture_boundary_weak` | 0.0173 | -0.1027 | 0.0300 | 0.0500 | 0.0500 | `true` | `causal_score_not_positive;high_fp_spillover_rate_too_high;non_active_regression_rate_too_high;estimated_precision_drop_too_high;non_active_fp_delta_too_high;high_confidence_fp_delta_too_high` |
| 2 | 25 | `candidate_policy_1_roi_texture` | 9 | `texture_boundary_weak` | -0.0778 | -0.0778 | 0.0240 | 0.0480 | 0.0480 | `true` | `causal_score_not_positive;high_fp_spillover_rate_too_high;non_active_regression_rate_too_high;estimated_precision_drop_too_high;non_active_fp_delta_too_high;high_confidence_fp_delta_too_high` |
| 1 | 40 | `candidate_policy_1_roi_texture` | 9 | `texture_boundary_weak` | -0.0789 | -0.0789 | 0.0240 | 0.0480 | 0.0480 | `true` | `causal_score_not_positive;high_fp_spillover_rate_too_high;non_active_regression_rate_too_high;estimated_precision_drop_too_high;non_active_fp_delta_too_high;high_confidence_fp_delta_too_high` |
| 2 | 45 | `candidate_policy_1_roi_texture` | 9 | `texture_boundary_weak` | -0.0903 | -0.0903 | 0.0270 | 0.0500 | 0.0480 | `true` | `causal_score_not_positive;high_fp_spillover_rate_too_high;non_active_regression_rate_too_high;estimated_precision_drop_too_high;non_active_fp_delta_too_high;high_confidence_fp_delta_too_high` |
| 2 | 35 | `candidate_policy_1_roi_texture` | 9 | `texture_boundary_weak` | -0.0907 | -0.0907 | 0.0270 | 0.0500 | 0.0480 | `true` | `causal_score_not_positive;high_fp_spillover_rate_too_high;non_active_regression_rate_too_high;estimated_precision_drop_too_high;non_active_fp_delta_too_high;high_confidence_fp_delta_too_high` |
| 2 | 30 | `candidate_policy_1_roi_texture` | 6 | `low_contrast_fn` | -0.0972 | -0.0972 | 0.0300 | 0.0500 | 0.0480 | `true` | `causal_score_not_positive;high_fp_spillover_rate_too_high;non_active_regression_rate_too_high;estimated_precision_drop_too_high;non_active_fp_delta_too_high;high_confidence_fp_delta_too_high` |
| 2 | 20 | `candidate_policy_1_roi_texture` | 7 | `texture_boundary_weak` | -0.1093 | -0.1093 | 0.0330 | 0.0500 | 0.0500 | `true` | `causal_score_not_positive;high_fp_spillover_rate_too_high;non_active_regression_rate_too_high;estimated_precision_drop_too_high;non_active_fp_delta_too_high;high_confidence_fp_delta_too_high` |
| 1 | 25 | `candidate_policy_1_roi_texture` | 6 | `weak_localization` | -0.1158 | -0.1158 | 0.0300 | 0.0500 | 0.0500 | `true` | `causal_score_not_positive;high_fp_spillover_rate_too_high;non_active_regression_rate_too_high;estimated_precision_drop_too_high;non_active_fp_delta_too_high;high_confidence_fp_delta_too_high` |

## Conclusions

1. Current precision-aware decision stack is over-conservative for paper-mode image augmentation coverage: `true`.
2. Current CP-CATF image augmentation is zero because every replayed image candidate is rejected under the current risk stack; the selected fallback is sampler_only, but sampler weighting is still pending dataloader support, so it becomes strict no-op in execution.
3. Safe executable image candidate under current gate: `false`.
4. Recommended next lever: `sampler_only implementation first; graded attenuation before image rerun`.
5. Continue training now: `false`.
6. Run seed1/seed2 now: `false`.
7. Modify gate then rerun seed0: `true` if the next change is a graded attenuation gate; otherwise implement sampler_only first.
8. Use current result as paper method result: `false`.

## Recommendation Options

- Scheme A, strict gate: defensible only if zero image augmentation is an acceptable conclusion and the method pivots to sampler_only.
- Scheme B, graded risk gate: recommended for candidates with positive causal score and medium FP risk; reduce probability/strength and top-m before rejecting.
- Scheme C, precision-aware attenuation: recommended for positive-benefit candidates with mild precision risk; keep one low-risk op and re-evaluate at the next feedback epoch.
- Scheme D, implement sampler_only: recommended if image augmentation remains unsafe after attenuation analysis.

## Artifacts

- JSON: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\cp_catf_decision_coverage_audit\reports\decision_coverage_audit.json`.
- Candidate records: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\cp_catf_decision_coverage_audit\decision_records.csv`.
- Seed coverage: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\cp_catf_decision_coverage_audit\seed_level_coverage.csv`.
- Rejection reasons: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\cp_catf_decision_coverage_audit\rejection_reason_summary.csv`.
