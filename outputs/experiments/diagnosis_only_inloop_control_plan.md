# Diagnosis-Only In-Loop Control 50ep Plan

Generated: 2026-06-01

## Purpose

The CATF-v2 multiseed result showed that seed 1 passed constraints even though `ROI applied=0` and `industrial samples augmented=0`. This makes it necessary to isolate whether the gain came from actual industrial augmentation or from the in-loop diagnosis/callback training path, RNG ordering, or other control-flow differences.

## Proposed Experiment

Run `scripts/train_yolo_default_with_inloop_feedback.py` for 50 epochs with:

- YOLO default augmentation enabled.
- `feedback_enabled=true`.
- `diagnosis_only=true`.
- `industrial_aug_enabled=false`.
- No ROI augmentation.
- No sample-aware routing.
- No policy-state mutation.
- No threshold mutation.
- `train images=2301`.
- No fixed augmented dataset.

Command shape:

```powershell
D:\Anaconda\envs\pytorch\python.exe scripts\train_yolo_default_with_inloop_feedback.py `
  --model yolo11n.pt `
  --data outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml `
  --epochs 50 `
  --imgsz 1024 `
  --batch 2 `
  --workers 0 `
  --device 0 `
  --seed <seed> `
  --project outputs/experiments `
  --run-id diagnosis_only_inloop_control_seed<seed>_50ep `
  --feedback-enabled `
  --diagnosis-only `
  --no-industrial-aug-enabled `
  --catf-version v2 `
  --class-aware-feedback `
  --threshold-calibration-report `
  --feedback-interval 5 `
  --feedback-start-epoch 5
```

## Difference From Clean Native YOLO Default

Clean native YOLO default uses `YOLO.train` with no feedback callback and no diagnosis artifacts. Diagnosis-only keeps YOLO default augmentation and training settings, but registers the diagnosis callback and writes policy/history reports without changing augmentation, thresholds, sampling, or policy state.

## Difference From CATF-v2

CATF-v2 runs the same diagnosis path but may update class policy, route samples, and apply ROI-aware industrial augmentation. Diagnosis-only disables those actions, so any metric shift relative to clean native cannot be attributed to industrial augmentation.

## Interpretation

If diagnosis-only also improves over clean native, CATF-v2 gains must be interpreted cautiously because callback/RNG/training-path effects may contribute to or dominate the observed change.

If diagnosis-only does not improve while CATF-v2 improves under the same seeds and constraints, that supports the claim that class-aware industrial augmentation and routing provide the incremental benefit.

## Current Status

This plan was not executed as a 50 epoch experiment. A 10 epoch smoke at `outputs/experiments/diagnosis_only_inloop_control_10ep_smoke/` confirmed diagnosis callback execution with industrial augmentation, ROI augmentation, and policy updates all disabled.
