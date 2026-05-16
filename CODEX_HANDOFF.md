# Codex Handoff

## Repository

- Path: `E:\TJGY\MinPaper\MyAutoAugument`
- Branch: `codex/sync-latest`
- Remote: `https://github.com/Yxiaoyang121/MyAutoAugument.git`

## Current Scope

The project is now a diagnosis-driven augmentation pipeline for industrial defect detection. Keep work centered on augmentation, tiling, diagnosis, proxy safety, short-training validation, and audit artifacts. Do not reframe this as YOLO backbone/neck/head redesign.

## Important Current Capabilities

- `scripts/build_yolo_tiled_dataset.py` builds a tiled YOLO dataset with default `tile_size=1024`, `overlap=0.2`, `min_visibility=0.3`, and `keep_empty_ratio=0.1`.
- `diagnosis.json` includes `diagnosis_vector` scores and per-score evidence.
- `policy_mapping.py` uses severity-score dynamic formulas for operator probability and strength.
- Proxy ranking combines `proxy_score` and `SafetyScore`; bbox rates below soft targets are penalties instead of automatic rejection.
- `copy_paste` policies produce `copy_paste_filter_audit.md/json` and debug visualizations.
- `strategy_memory.py` appends short-training records to `outputs/strategy_memory.jsonl` and reranks candidates with cosine similarity when prior cases exist.
- `metric_consistency_audit.md` documents differences between YOLO val metrics and diagnosis TP/FP/FN.

## Key Files

- `scripts/build_yolo_tiled_dataset.py`
- `scripts/run_diagnostic_augmentation_pipeline.py`
- `AutoAugment/diagnostic_pipeline/diagnosis.py`
- `AutoAugment/diagnostic_pipeline/policy_mapping.py`
- `AutoAugment/diagnostic_pipeline/proxy_evaluation.py`
- `AutoAugment/diagnostic_pipeline/strategy_memory.py`
- `AutoAugment/diagnostic_pipeline/metric_audit.py`
- `AutoAugment/augmentations/ops.py`
- `AutoAugment/search/proxy_metrics.py`
- `AutoAugment/search/evaluator.py`

## Latest Verified Outputs

- Tiled dataset: `outputs/tiled_dataset_smoke`
- Tiled smoke pipeline: `outputs/diagnostic_aug_tiled_smoke`
- Copy-paste audit: `outputs/diagnostic_aug_tiled_smoke/proxy/copy_paste_filter_audit.md`
- Metric audit: `outputs/diagnostic_aug_tiled_smoke/metric_consistency/metric_consistency_audit.md`
- Memory report: `outputs/diagnostic_aug_tiled_smoke/strategy_memory/strategy_memory_report.md`

## Verification Done

- `pytest -q tests/test_build_yolo_tiled_dataset.py tests/test_copy_paste.py tests/test_proxy_prefilter.py tests/test_yolo_error_analysis.py`
- `pytest -q`
- `python scripts\build_yolo_tiled_dataset.py --dataset-root E:\TJGY\DataSet2_fixed --data-yaml E:\TJGY\DataSet2_fixed\data.yaml --output-dir outputs\tiled_dataset_smoke --tile-size 1024 --overlap 0.2 --min-visibility 0.3 --keep-empty-ratio 0.1 --max-images-per-split 8 --debug-limit 8 --overwrite`
- `python scripts\run_diagnostic_augmentation_pipeline.py --dataset-root outputs\tiled_dataset_smoke --data-yaml outputs\tiled_dataset_smoke\data.yaml --output-dir outputs\diagnostic_aug_tiled_smoke --model yolo11n.pt --imgsz 640 --batch 4 --baseline-epochs 1 --short-epochs 1 --final-epochs 1 --top-k 1 --workers 0 --device cpu --skip-final-train --proxy-samples 32 --augment-repeat 1`

## Notes

- The YOLO smoke required `KMP_DUPLICATE_LIB_OK=TRUE` in this Anaconda CPU environment due duplicate OpenMP runtime initialization.
- The smoke is intentionally capped and is not a benchmark result.
- Output directories are ignored by `.gitignore`, but `outputs/project_snapshot_latest.md` is tracked.
