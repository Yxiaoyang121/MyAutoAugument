# Project Snapshot

- Generated: 2026-05-17T18:06:58
- Branch: codex/sync-latest
- Commit: 072e583c85de6120d0e35178a29857eb3489b52d
- Remote: https://github.com/Yxiaoyang121/MyAutoAugument.git

## Working Tree

```text
 M CODEX_HANDOFF.md
 M EXPERIMENT_LOG.md
 M PROJECT_STATE.md
 M outputs/project_snapshot_latest.md
```

## Key Files

- README.md
- PROJECT_STATE.md
- CODEX_HANDOFF.md
- EXPERIMENT_LOG.md
- AGENTS.md
- docs/ARCHITECTURE_CURRENT.md
- docs/diagnostic_augmentation_framework.md
- docs/experiment_protocol.md
- scripts/build_yolo_tiled_dataset.py
- scripts/run_gpu_preflight.py
- scripts/run_diagnostic_augmentation_pipeline.py
- outputs/gpu_preflight_report.md
- outputs/gpu_preflight_report.json
- outputs/tiled_baseline_20epoch/baseline_20epoch_report.md
- outputs/tiled_baseline_20epoch/baseline_20epoch_metrics.json
- AutoAugment/diagnostic_pipeline/__init__.py
- AutoAugment/diagnostic_pipeline/strategy_memory.py
- AutoAugment/diagnostic_pipeline/metric_audit.py
- AutoAugment/diagnostic_pipeline/proxy_evaluation.py
- AutoAugment/diagnostic_pipeline/policy_mapping.py

## Tracked File Count

- 115 tracked files

## Notes

- Tiled baseline 20 epoch completed in conda env `pytorch`.
- Formal training Python: `D:\Anaconda\envs\pytorch\python.exe`.
- Training settings: `model=yolo11n.pt epochs=20 imgsz=1024 batch=2 workers=0 device=0`.
- YOLO built-in augmentations were disabled with the requested zero-valued knobs.
- CUDA OOM occurred: false.
- Tiled baseline metrics: Precision 0.828, Recall 0.213, mAP50 0.247, mAP50-95 0.181.
- best.pt: `outputs/tiled_baseline_20epoch/train/weights/best.pt`.
- CPU remains smoke/debug only; formal training uses GPU `device=0`.
- This snapshot reflects the current local repository state.
- It does not invent benchmark results.
