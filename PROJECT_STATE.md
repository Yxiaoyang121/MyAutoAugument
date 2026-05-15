# Project State

Last updated: 2026-05-15

## Current Position

This repository now centers on a validation-error diagnostic-driven augmentation framework for industrial defect detection.

The active implementation includes:

- `AutoAugment/diagnostic_pipeline/` for the end-to-end closed loop
- `scripts/run_diagnostic_augmentation_pipeline.py` as the unified entry point
- updated README and method/protocol documentation
- a cleanup fix in `AutoAugment/search/random_search.py` to keep the existing proxy-prefilter tests green on Windows

## What Is Implemented

- baseline YOLO training command capture and audit logs
- validation prediction capture with per-image GT/prediction records
- validation error diagnosis and `diagnosis.json`
- diagnosis-to-policy mapping to `candidate_policies.json`
- proxy evaluation and ranking
- short-training selection records and `selected_policy.json`
- final augmented dataset builder
- final training/reporting scaffolding

## What Is Still Deferred

- full large-scale smoke or benchmark training on a real dataset
- fixed/random augment comparison runs
- paper tables with final numeric results

## Verified Locally

- Python syntax checks on the new pipeline modules passed
- `pytest -q tests/test_yolo_error_analysis.py tests/test_proxy_prefilter.py tests/test_yolo_train_evaluator.py` passed
- a dry-run of the pipeline entrypoint completed successfully

## Notes

- Windows YOLO commands default to `workers=0`
- the current pipeline keeps outputs auditable through JSON, Markdown, and log files
