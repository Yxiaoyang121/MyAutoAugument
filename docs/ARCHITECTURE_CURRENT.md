# Current Architecture

## Overall Shape

The repository has two layers:

1. reusable augmentation, bbox, YOLO dataset, and evaluator utilities
2. a diagnosis-driven augmentation pipeline for industrial defect detection

The project does not modify YOLO backbone, neck, or head architecture.

## Reusable Layer

- `AutoAugment/augmentations/` registers image and bbox-aware augmentations.
- `AutoAugment/augmentations/ops.py` includes same-image bbox-level `copy_paste`.
- `AutoAugment/bbox/` handles bbox conversion, clipping, affine transforms, and IoU.
- `AutoAugment/formats/` and `AutoAugment/utils/yolo_dataset.py` load and write YOLO-format datasets.
- `AutoAugment/policies/` defines `Policy` and `OperationSpec`.
- `AutoAugment/search/` keeps proxy metrics, evaluator adapters, and legacy random-search behavior.
- `AutoAugment/diagnostics/` performs validation error analysis and heuristic advice generation.

## Dataset Construction

- `scripts/build_yolo_tiled_dataset.py` creates tiled YOLO datasets for large images.
- Defaults are `tile_size=1024`, `overlap=0.2`, `min_visibility=0.3`, and `keep_empty_ratio=0.1`.
- Outputs include tiled `images/train`, `images/val`, `labels/train`, `labels/val`, `data.yaml`, `tiled_dataset_report.md/json`, and `debug_tiling/` visualizations.

## Diagnosis-Driven Pipeline

Modules under `AutoAugment/diagnostic_pipeline/`:

- `baseline.py`: baseline YOLO train/val command capture.
- `prediction.py`: validation prediction capture.
- `diagnosis.py`: error diagnosis plus normalized `diagnosis_vector`.
- `policy_mapping.py`: severity-score dynamic policy generation with runtime op validation.
- `proxy_evaluation.py`: proxy scoring, safety scoring, copy-paste audit, and ranking.
- `strategy_memory.py`: JSONL strategy memory and cosine-similarity reranking.
- `short_training.py`: top-k short training selector.
- `dataset_builder.py`: final augmented dataset construction.
- `final_training.py`: final train/val stage.
- `metric_audit.py`: YOLO val metric versus diagnosis TP/FP/FN consistency audit.
- `reporting.py`: experiment summaries.
- `common.py`: JSON, Markdown, command, and log helpers.

## Ranking Flow

1. Baseline YOLO training and validation produce model metrics.
2. Validation prediction labels are analyzed into TP/FP/FN, quality, size, class, and position summaries.
3. `diagnosis_vector` converts dominant failure modes into normalized severity scores.
4. `policy_mapping.py` converts severities into operator probability and strength formulas.
5. Proxy evaluation computes `proxy_score` and `SafetyScore`.
6. Hard reject is reserved for severe bbox/class/exposure errors; ordinary risk is recorded as `safety_soft_penalty_reasons`.
7. Strategy memory optionally boosts candidates similar to historically successful diagnosis cases.
8. Top-k policies run short training and the best candidate builds the final augmented dataset.

## Output Conventions

The normalized output layout is documented in `docs/output_convention.md`.
Active artifacts now use these roots:

- `outputs/datasets/` for generated datasets.
- `outputs/experiments/` for training experiment results.
- `outputs/audits/` for preflight checks, diagnostics, and inventory reports.
- `outputs/snapshots/` for project snapshots.
- `outputs/archive/` for historical artifacts moved out of active paths.

Current active paths:

- Tiled smoke dataset: `outputs/datasets/tiled/tiled_1024_ov20_smoke/`
- Future full tiled dataset: `outputs/datasets/tiled/tiled_1024_ov20_full/`
- Current tiled baseline: `outputs/experiments/20260517_tiled_baseline_20epoch/`
- GPU preflight: `outputs/audits/gpu_preflight/`
- Artifact inventory: `outputs/audits/artifact_inventory/`

Each stage writes auditable artifacts under the selected output directory:

- command text
- stdout and stderr logs
- JSON payloads
- Markdown summaries
- debug visualizations for tiling and copy-paste where applicable

Formal YOLO commands must explicitly set `project=outputs/experiments/<run_id>`.
Ultralytics auto outputs under `runs/detect/*` are not formal records and must be
migrated or archived. Historical root-level outputs were archived to
`outputs/archive/old_outputs_20260517/`; historical `runs/detect/*` outputs were
archived to `outputs/archive/old_runs_20260517/runs_detect/`.

## Operational Constraints

- Formal training must use conda env `pytorch` and `D:\Anaconda\envs\pytorch\python.exe`.
- Formal training must use GPU `device=0`; CPU is only for smoke/debug.
- YOLO commands should default to `workers=0` on Windows.
- Dry-run mode must not start training.
- This environment may require `KMP_DUPLICATE_LIB_OK=TRUE` for CPU YOLO runs due duplicate OpenMP runtime initialization.
- Smoke outputs are not benchmark claims.
