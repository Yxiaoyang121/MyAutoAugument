# CP-CATF Precision Gate Dry Run - Seed0

## Scope

- No training was run.
- The dry run replays epoch 25 from the existing paper-mode seed0 execution-fixed run.
- Policy selection source is the paper-mode probe split.
- Final validation is not used for candidate decision.

## Original Epoch 25

- Candidate: `candidate_policy_1_roi_texture`.
- Original accepted: `true`.
- Active class: `9`.
- Ops: `sharpen_mild, local_contrast`.
- ROI applied in original run: `312`.
- Industrial image augmented in original run: `226`.
- Router random draw count in original run: `1330`.

## New Precision-Aware Decision

- ROI texture decision: `reject`.
- Image modification allowed: `false`.
- Rejection reasons: `causal_score_not_positive, high_fp_spillover_rate_too_high, non_active_regression_rate_too_high, estimated_precision_drop_too_high, non_active_fp_delta_too_high, high_confidence_fp_delta_too_high`.
- estimated_precision_drop: `0.0300`.
- non_active_fp_delta: `0.0500`.
- high_confidence_fp_delta: `0.0500`.
- Selected candidate after replay: `candidate_policy_3_sampler_only` / `sampler_only`.
- Sampler-only effective: `false`.
- Strict no-op if sampler pending: `true`.

## Leakage And Specificity

- Final val leakage: `false`.
- Decision depends on seed id: `false`.
- Decision depends on fixed class id: `false`.
- RiskGuard used as final rule: `false`.

## Recommendation

- Recommend seed0 50ep rerun: `true`.
