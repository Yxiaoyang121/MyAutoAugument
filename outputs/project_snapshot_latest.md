# Project Snapshot

- Generated: 2026-05-17T13:39:27
- Branch: codex/sync-latest
- Commit: c5c2e0904eee6661b821ca2194ed690e451e000a
- Remote: https://github.com/Yxiaoyang121/MyAutoAugument.git

## Working Tree

```text
 M CODEX_HANDOFF.md
 M EXPERIMENT_LOG.md
 M PROJECT_STATE.md
 M outputs/gpu_preflight_report.json
 M outputs/gpu_preflight_report.md
 M outputs/project_snapshot_latest.md
 M scripts/run_gpu_preflight.py
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
- AutoAugment/diagnostic_pipeline/__init__.py
- AutoAugment/diagnostic_pipeline/strategy_memory.py
- AutoAugment/diagnostic_pipeline/metric_audit.py
- AutoAugment/diagnostic_pipeline/proxy_evaluation.py
- AutoAugment/diagnostic_pipeline/policy_mapping.py

## Tracked File Count

- 115 tracked files

## Notes

- Current GPU preflight passed in conda env `pytorch`.
- Current training Python: `D:\Anaconda\envs\pytorch\python.exe`.
- Current training stack: PyTorch 2.4.1, CUDA 12.4, Ultralytics 8.3.221.
- Current GPU: NVIDIA GeForce RTX 3060 Laptop GPU, CUDA device count 1.
- Minimal YOLO GPU smoke passed with `device=0`, `workers=0`, and YOLO built-in augmentations disabled.
- Formal training must use conda env `pytorch` and YOLO `device=0`; CPU is only for smoke/debug.
- Do not use base for formal training.
- This snapshot reflects the current local repository state.
- It does not invent benchmark results.
