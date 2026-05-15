# AGENTS.md

## Repo Rules For Future Agents

- Keep the project centered on diagnosis-driven augmentation, not YOLO network redesign.
- Preserve existing CLI behavior unless a new entrypoint is explicitly added.
- Use `pathlib.Path` for paths and keep outputs auditable.
- Do not start large training runs by default.
- Default YOLO `workers=0` on Windows.
- Reuse the existing augmentation, bbox, dataset, and evaluator code when possible.

## Preferred Working Pattern

- inspect first
- keep changes scoped
- add new modules instead of large renames
- verify with syntax checks and targeted tests
- record commands, logs, and JSON outputs for each stage

## High-Signal Files

- `README.md`
- `AutoAugment/diagnostic_pipeline/`
- `scripts/run_diagnostic_augmentation_pipeline.py`
- `docs/diagnostic_augmentation_framework.md`
- `docs/experiment_protocol.md`

## Current Status Summary

- baseline and diagnostic pipeline scaffolding are present
- regression tests for the old search/diagnostic path are passing
- dry-run pipeline output exists under `outputs/diagnostic_aug_pipeline_smoke`
