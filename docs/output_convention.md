# Output Convention

This project keeps source code and generated artifacts separate. Formal experiment
results must not be scattered under `outputs/` root or `runs/detect`.

## Directory Layout

```text
outputs/
  datasets/
    tiled/
      tiled_1024_ov20_smoke/
      tiled_1024_ov20_full/

  experiments/
    20260517_tiled_baseline_20epoch/
      configs/
      logs/
      reports/
      train/
      val/
      weights/
      artifacts/

  audits/
    gpu_preflight/
    class_distribution/
    metric_consistency/
    copy_paste/
    artifact_inventory/

  snapshots/
    project_snapshot_latest.md

  archive/
    old_outputs_20260517/
    old_runs_20260517/
```

## Rules

- `outputs/datasets/` is only for generated datasets.
- `outputs/experiments/` is only for training experiment results.
- `outputs/audits/` is only for checks, diagnostics, and audit reports.
- `outputs/archive/` is only for historical artifacts moved out of active paths.
- `runs/` is not a formal result location.
- Ultralytics auto outputs under `runs/detect/*` must be migrated into
  `outputs/experiments/<run_id>/` or archived under `outputs/archive/`.
- Every formal training run must specify `project=outputs/experiments/<run_id>`.
- Every formal training run must record a `run_id`, dataset path, `data.yaml`,
  train/val commands, environment, weights paths, metrics, and augmentation
  configuration.
- Large weights, generated datasets, images, and archived outputs are not
  committed to GitHub by default.

## Current Active Paths

- Smoke tiled dataset:
  `outputs/datasets/tiled/tiled_1024_ov20_smoke/`
- Future full tiled dataset:
  `outputs/datasets/tiled/tiled_1024_ov20_full/`
- Current tiled baseline:
  `outputs/experiments/20260517_tiled_baseline_20epoch/`
- GPU preflight audit:
  `outputs/audits/gpu_preflight/`
- Artifact inventory:
  `outputs/audits/artifact_inventory/`
- Project snapshot:
  `outputs/snapshots/project_snapshot_latest.md`

The compatibility snapshot path `outputs/project_snapshot_latest.md` is retained.

## Cleanup Status On 2026-05-17

- Legacy root-level outputs were archived to `outputs/archive/old_outputs_20260517/`.
- Legacy Ultralytics outputs from `runs/detect/*` were archived to
  `outputs/archive/old_runs_20260517/runs_detect/`.
- `runs/detect` is currently empty and must stay non-authoritative.
- The current tiled baseline is under
  `outputs/experiments/20260517_tiled_baseline_20epoch/`.
- The current tiled smoke dataset is under
  `outputs/datasets/tiled/tiled_1024_ov20_smoke/`.
- GPU preflight reports are under `outputs/audits/gpu_preflight/`.

## Git Tracking Policy

- Do not commit weights such as `*.pt`.
- Do not commit generated dataset images or labels.
- Do not commit generated train/val plots or batch images.
- Do not commit `outputs/archive/`.
- Commit lightweight configs, commands, logs, Markdown reports, JSON reports, and snapshots when they document an experiment or audit.
