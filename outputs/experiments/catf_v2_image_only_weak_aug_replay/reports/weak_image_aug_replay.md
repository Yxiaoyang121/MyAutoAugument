# Weak Image Augmentation Replay

## Scope

- No training was run.
- Replay is image-only: `candidate_policy_3_sampler_only` is disabled for the main path.
- No weighted index list, sampler-only, or training-sampling intervention participates in decisions.
- Candidate policy added for replay: `candidate_policy_1b_weak_roi_texture`.
- `candidate_policy_1b_weak_roi_texture` keeps one low-risk op, reduces prob/strength, caps samples per feedback interval, and still must pass precision and non-active regression gates.

## Replay Rules

- Original policy: `candidate_policy_1_roi_texture` with `sharpen_mild` + `local_contrast`.
- Weak policy: keep only one op. The replay chooses `local_contrast` when the same-epoch low-contrast image proxy is available; otherwise it uses a generic op-risk prior.
- Tested attenuation ratios: `0.5` and `0.25`.
- `prob_multiplier = attenuation_ratio` and `strength_multiplier = attenuation_ratio`.
- Max augmented samples per feedback interval: `64 * attenuation_ratio`.
- Risk is attenuated as `proxy_or_original_risk * attenuation_ratio * single_op_risk_factor`.
- The replay uses no seed-id, class-id, or dataset-class-name hard rule.

## Coverage

- Total image candidates: `54`.
- Original ROI texture candidates: `27`.
- Legacy image-evidence candidates: `8`.
- Weak ROI texture candidates accepted by replay: `8`.
- Strict no-op decisions: `19`.
- Ratio 0.5 accepts: `0`.
- Ratio 0.25 accepts: `8`.
- All no-op: `false`.
- Final-val leakage: `false`.
- Sampler-only involved: `false`.

## Seed-Level Decisions

| seed | ROI texture candidates | weak_roi_texture | strict_noop | weak epochs |
|---:|---:|---:|---:|---|
| 0 | 9 | 1 | 8 | [25] |
| 1 | 9 | 2 | 7 | [25, 40] |
| 2 | 9 | 5 | 4 | [20, 25, 30, 35, 45] |

## Original Rejection Reasons

| reason | count |
|---|---:|
| causal_score_not_positive | 27 |
| high_fp_spillover_rate_too_high | 20 |
| non_active_regression_rate_too_high | 20 |
| estimated_precision_drop_too_high | 20 |
| non_active_fp_delta_too_high | 20 |
| high_confidence_fp_delta_too_high | 20 |
| insufficient_active_class_benefit | 14 |
| insufficient_evidence_count | 7 |
| diagnosis_confidence_too_low | 7 |
| bbox_instability_rate_too_high | 5 |

## Answers

- Weak image augmentation replay completed: `true`.
- Safe weak candidate count: `8`.
- Seed2 has offline gate-safe weak image candidate: `true`.
- Seed0/seed1 legacy image candidates preserved as weak image candidates: `true`.
- Sampler-only completely excluded: `true`.
- Final-val leakage false: `true`.
- Avoids fixed seed/class special case: `true`.
- Recommend seed2 50ep image-only validation: `true`.
- Recommended run order: `seed2_first_then_seed0_seed1_sanity`.

## Interpretation

Ratio `0.5` remains too risky under the non-active and precision gates. Ratio `0.25` produces gate-safe weak ROI texture candidates for the legacy image-evidence rows while leaving the remaining ROI texture rows as strict no-op.

Seed2 has offline gate-safe weak candidates, but this is not a training result. Because fixed CATF-v2 seed2 failed through high-risk image augmentation and non-active regression, the next validation should be seed2-only image augmentation training before any seed0/seed1 sanity runs.
