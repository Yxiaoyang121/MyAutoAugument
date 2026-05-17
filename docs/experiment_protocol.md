# Experiment Protocol

## Scope

This project compares data augmentation strategies inside a fixed YOLO training framework. Backbone, neck, and head stay unchanged. The research variable is diagnosis-driven augmentation policy selection, not network redesign.

## Environment Rules

- Formal training must use conda env `pytorch`.
- Formal training Python: `D:\Anaconda\envs\pytorch\python.exe`.
- Formal YOLO training must use `device=0`.
- Windows YOLO runs must use `workers=0`.
- CPU is allowed only for smoke/debug and must not be reported as formal experiment output.
- The base conda environment must not be used for formal training because it previously resolved to CPU-only PyTorch.

## Artifact Layout

Follow `docs/output_convention.md`.

- Generated datasets: `outputs/datasets/`
- Formal experiments: `outputs/experiments/<run_id>/`
- Audits and diagnostics: `outputs/audits/`
- Snapshots: `outputs/snapshots/`
- Archive: `outputs/archive/`
- `runs/` is not a formal result location.

Every formal YOLO command must explicitly set `project=outputs/experiments/<run_id>`. Do not rely on the Ultralytics default `runs/detect` location.

## Active Dataset Paths

- Smoke tiled dataset: `outputs/datasets/tiled/tiled_1024_ov20_smoke/`
- Future full tiled dataset: `outputs/datasets/tiled/tiled_1024_ov20_full/`

The current smoke tiled dataset used `--max-images-per-split 8`, so it must not be treated as a full benchmark dataset.

## Experiment Groups

1. Baseline: original data, fixed YOLO model, no external diagnostic augmentation.
2. Fixed Augment: manually selected augmentation policy.
3. Random Augment: random augmentation policy search.
4. Ours: diagnosis-driven augmentation policy selection.

## Ablations

1. Full method.
2. Without diagnosis: generate policies without validation-error diagnosis.
3. Without proxy filter: skip proxy safety and quality filtering.
4. Without short training: select from proxy ranking only.
5. Fixed strong augmentation.
6. Fixed weak augmentation.

## Metrics

- mAP50
- mAP50-95
- Precision
- Recall
- small-object recall when available
- false positives and false negatives when available
- training time
- GPU memory when available
- policy-search cost

## Required Records Per Experiment

Each experiment must record:

- `run_id`
- conda environment and `sys.executable`
- PyTorch, CUDA, GPU, and Ultralytics versions
- dataset path and `data.yaml`
- model weights input
- train command and val command
- stdout and stderr logs
- epoch, imgsz, batch, workers, and device
- YOLO built-in augmentation config
- external augmentation config
- `best.pt` and `last.pt` local paths
- metrics JSON and Markdown summary
- diagnosis files
- policy files
- proxy evaluation files
- short-training records when used

Large weights, generated images, dataset image/label files, and archive contents are local artifacts and are ignored by Git by default. Lightweight reports, summaries, configs, logs, and snapshots can be tracked.

## Recommended Execution Order

1. Build or verify the target dataset.
2. Run GPU preflight in conda env `pytorch`.
3. Run baseline YOLO training and validation.
4. Save baseline metrics, `best.pt`, `last.pt`, commands, logs, and environment metadata.
5. Run validation prediction capture using baseline `best.pt`.
6. Generate `diagnosis.json` and `diagnosis_summary.md`.
7. Generate `candidate_policies.json`.
8. Run proxy evaluation and save `proxy_metrics.json` and `proxy_ranking.json`.
9. Run short training for top-k policies.
10. Select the best candidate policy.
11. Build the final augmented dataset.
12. Run final YOLO training and validation.
13. Export `experiment_summary.md`, `experiment_summary.json`, and paper tables.

## Current Baseline Note

The current `20260517_tiled_baseline_20epoch` result is useful as a GPU and pipeline check, but it is not a formal final result because it uses `outputs/datasets/tiled/tiled_1024_ov20_smoke/`. Before formal comparison, build and audit `outputs/datasets/tiled/tiled_1024_ov20_full/`.
