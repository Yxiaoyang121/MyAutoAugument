# Current Architecture

## Overall Shape

The repository is organized around two layers:

1. reusable augmentation and YOLO utilities
2. a diagnosis-driven augmentation pipeline for defect detection

The codebase is not a YOLO architecture modification project. The YOLO backbone, neck, and head remain unchanged.

## Reusable Layer

- `AutoAugment/augmentations/` registers and applies image augmentations
- `AutoAugment/bbox/` handles bbox conversion, clipping, affine transforms, and IoU
- `AutoAugment/datasets/` and `AutoAugment/utils/yolo_dataset.py` load and write YOLO-format datasets
- `AutoAugment/policies/` defines `Policy`, `OperationSpec`, and search spaces
- `AutoAugment/search/` keeps the existing search/evaluator/proxy machinery
- `AutoAugment/diagnostics/` contains validation error analysis and heuristic advice

## Diagnosis-Driven Pipeline Layer

New modules live in `AutoAugment/diagnostic_pipeline/`:

- `baseline.py`: baseline YOLO training and validation command audit
- `prediction.py`: validation prediction capture and per-image records
- `diagnosis.py`: error diagnosis and stable `diagnosis.json`
- `policy_mapping.py`: diagnosis-to-policy mapping
- `proxy_evaluation.py`: no-training proxy metrics and ranking
- `short_training.py`: Top-K short training selector
- `dataset_builder.py`: final augmented dataset construction
- `final_training.py`: final YOLO train/val and report
- `reporting.py`: experiment summary and paper tables
- `common.py`: logging, JSON, markdown, and command helpers

## Entry Points

- `scripts/run_diagnostic_augmentation_pipeline.py`
- existing legacy search entrypoints remain in `examples/` and `tools/`

## Output Conventions

Every stage writes auditable artifacts under the chosen output directory:

- command text
- stdout and stderr logs
- stage JSON payloads
- Markdown summaries where applicable

## Known Operational Constraints

- YOLO commands default to `workers=0` on Windows
- dry-run mode must not trigger full training
- final benchmark results are not yet produced in this snapshot
