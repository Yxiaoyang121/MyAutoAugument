# Seed0 Precision-Aware CP-CATF Gate Update

## Scope

This update does not run training, seed1, seed2, or multiseed validation. It converts the seed0 precision-risk audit into a generic CP-CATF accept-gate change.

## Motivation

The paper-mode seed0 execution-fixed run proved the accept-to-execution path:

- `candidate_policy_1_roi_texture` was accepted at epoch 25.
- Executable policy was generated.
- Sample router was called.
- ROI applied = 312.
- Industrial image augmented = 226.
- Router random draw count = 1330.
- Final validation leakage = false.

The run failed the industrial constraint only because Precision dropped by more than 0.01:

- clean paper seed0 Precision = 0.7513.
- CP-CATF seed0 Precision = 0.7399.
- Delta Precision = -0.0114.

The precision-risk audit found that the dominant issue was non-active false-positive spillover, not class 9 itself. Class 9 improved locally, while non-active classes such as class 5, class 6, class 8, and class 2 drove the operating-point Precision loss.

## Implementation

The CP-CATF gate now records and enforces three precision-risk fields before accepting image-modifying candidates:

- `estimated_precision_drop`, rejected when above 0.005.
- `non_active_fp_delta`, rejected when above 0.005.
- `high_confidence_fp_delta`, rejected when above 0.0.

These fields are reject-only gate terms. They do not change the causal score formula.

Paper-mode risk estimation now uses the full probe-split per-class context when estimating non-active risk. Candidate selection still uses active rows, but risk evaluation can see non-active classes.

## Expected Behavior

An image candidate can still pass when probe evidence shows active-class benefit and low global risk. A seed0-like candidate with non-active FP pressure is rejected before training image augmentation, unless the probe split estimates the Precision risk within the configured margins.

This remains dataset-agnostic:

- no seed-id rule;
- no fixed class-id rule;
- no class-name rule;
- RiskGuard remains audit/debug prior only.

## Verification

- `python -m py_compile AutoAugment/catf_v2/causal_probe.py scripts/train_yolo_default_with_inloop_feedback.py AutoAugment/catf_v2/sample_router.py AutoAugment/catf_v2/policy_matrix.py`
- `pytest -q tests/test_cp_catf_accept_to_execution.py tests/test_cp_catf_paper_mode.py tests/test_catf_v2_causal_probe.py tests/test_catf_v2_transform_bypass.py tests/test_catf_v2_policy_matrix.py tests/test_catf_v2_sample_router.py tests/test_catf_v2_roi_augmentation.py tests/test_inloop_feedback_training.py tests/test_online_augmentation.py`
- Result: `70 passed`.

## Next Step

Do not resume seed1/seed2 or multiseed training until a paper-mode probe dry run confirms the new precision-aware gate gives the intended candidate decisions.
