# Experiment Log

## 2026-05-15

### Code Refactor

- added the diagnosis-driven augmentation pipeline modules under `AutoAugment/diagnostic_pipeline/`
- added a unified pipeline entrypoint at `scripts/run_diagnostic_augmentation_pipeline.py`
- updated documentation and README to reflect the new research framing
- fixed the proxy-prefilter cleanup path in `AutoAugment/search/random_search.py`

### Verification

- syntax check on all new pipeline modules passed
- regression tests passed:
  - `pytest -q tests/test_yolo_error_analysis.py tests/test_proxy_prefilter.py tests/test_yolo_train_evaluator.py`
- dry-run pipeline invocation passed:
  - `python scripts/run_diagnostic_augmentation_pipeline.py --dataset-root outputs\\dryrun_dataset --data-yaml outputs\\dryrun_dataset\\data.yaml --output-dir outputs\\diagnostic_aug_pipeline_smoke --model yolo11n.pt --imgsz 640 --batch 4 --baseline-epochs 5 --short-epochs 3 --final-epochs 5 --top-k 3 --workers 0 --device 0 --dry-run --skip-final-train`

### Output Artifact

- generated `outputs/diagnostic_aug_pipeline_smoke/pipeline_summary.json`
- generated `outputs/diagnostic_aug_pipeline_smoke/report/experiment_summary.md`

### Notes

- no full training run was executed
- no benchmark metrics are recorded here
