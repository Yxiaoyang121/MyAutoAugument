# Codex Handoff

## Repository

- Path: `E:\TJGY\MinPaper\MyAutoAugument`
- Remote: `https://github.com/Yxiaoyang121/MyAutoAugument.git`

## Working Tree Intent

The repository has been refocused from random augmentation search toward a diagnosis-driven augmentation optimization framework for industrial defect detection.

## Important Changes In The Current State

- added `AutoAugment/diagnostic_pipeline/` with baseline, prediction, diagnosis, policy mapping, proxy evaluation, short training, final dataset building, final training, and reporting modules
- added `scripts/run_diagnostic_augmentation_pipeline.py`
- added `docs/diagnostic_augmentation_framework.md`
- added `docs/experiment_protocol.md`
- updated `README.md` to reflect the new positioning
- fixed proxy-prefilter cleanup behavior in `AutoAugment/search/random_search.py`

## Behavior To Preserve

- existing CLI behavior in `examples/run_policy_search.py`
- existing augmentation operators, bbox transforms, and YOLO dataset I/O
- Windows default `workers=0` for YOLO calls
- auditability of every trial and pipeline stage

## Files Worth Reading First

- `AutoAugment/diagnostic_pipeline/common.py`
- `AutoAugment/diagnostic_pipeline/diagnosis.py`
- `AutoAugment/diagnostic_pipeline/policy_mapping.py`
- `AutoAugment/diagnostic_pipeline/proxy_evaluation.py`
- `AutoAugment/diagnostic_pipeline/short_training.py`
- `scripts/run_diagnostic_augmentation_pipeline.py`
- `docs/diagnostic_augmentation_framework.md`

## Local Verification Done

- syntax checked the new modules
- ran the existing regression tests for YOLO error analysis, proxy prefilter, and train/val evaluator
- ran a dry-run pipeline invocation that produced `outputs/diagnostic_aug_pipeline_smoke`

## Caution

- do not claim final benchmark numbers from this tree; only dry-run and test results are currently confirmed
