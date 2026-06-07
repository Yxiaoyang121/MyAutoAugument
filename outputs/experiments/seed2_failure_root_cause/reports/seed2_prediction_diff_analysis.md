# Seed2 Prediction Diff Analysis

This report uses prediction-only runs on existing clean/fixed best weights. No training was run.

- Clean prediction JSON: `outputs/experiments/seed2_failure_root_cause/predictions/clean_best/validation_predictions.json`.
- Fixed prediction JSON: `outputs/experiments/seed2_failure_root_cause/predictions/fixed_best/validation_predictions.json`.
- Clean prediction count: `1041`.
- Fixed prediction count: `1097`.
- Delta fixed-clean prediction count: `56`.

## Error Buckets

- Clean detected, fixed missed: `24` GT objects.
- Clean high-IoU, fixed worse localization: `11` GT objects.
- Fixed confidence drop on matched objects: `20` GT objects.
- Fixed new high-confidence FP: `182` predictions.

## Class Confidence Count Changes

| class | clean pred count | fixed pred count | delta count | clean mean conf | fixed mean conf | delta mean conf |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 84 | 86 | +2 | 0.8751 | 0.8784 | +0.0033 |
| 1 | 337 | 339 | +2 | 0.8384 | 0.8357 | -0.0027 |
| 2 | 9 | 7 | -2 | 0.7677 | 0.7421 | -0.0255 |
| 3 | 17 | 7 | -10 | 0.5186 | 0.6402 | +0.1216 |
| 4 | 28 | 33 | +5 | 0.6074 | 0.5200 | -0.0874 |
| 5 | 25 | 10 | -15 | 0.5173 | 0.5104 | -0.0069 |
| 6 | 57 | 59 | +2 | 0.6848 | 0.6788 | -0.0061 |
| 7 | 269 | 319 | +50 | 0.6300 | 0.6613 | +0.0313 |
| 8 | 40 | 64 | +24 | 0.7535 | 0.6210 | -0.1325 |
| 9 | 91 | 83 | -8 | 0.7155 | 0.6606 | -0.0549 |
| 10 | 15 | 13 | -2 | 0.8345 | 0.7960 | -0.0384 |
| 11 | 26 | 28 | +2 | 0.8388 | 0.7693 | -0.0694 |
| 12 | 43 | 49 | +6 | 0.7664 | 0.7370 | -0.0295 |

## Class Best-IoU Changes

| class | GT count | clean mean best IoU | fixed mean best IoU | delta |
|---|---:|---:|---:|---:|
| 0 | 64 | 0.8985 | 0.9070 | +0.0085 |
| 1 | 274 | 0.9017 | 0.8974 | -0.0043 |
| 2 | 8 | 0.6710 | 0.6651 | -0.0059 |
| 3 | 2 | 0.9478 | 0.9388 | -0.0090 |
| 4 | 18 | 0.2222 | 0.4151 | +0.1929 |
| 5 | 13 | 0.4968 | 0.5019 | +0.0051 |
| 6 | 46 | 0.6142 | 0.5875 | -0.0267 |
| 7 | 269 | 0.5695 | 0.6142 | +0.0447 |
| 8 | 42 | 0.5087 | 0.5306 | +0.0219 |
| 9 | 84 | 0.5436 | 0.4792 | -0.0644 |
| 10 | 12 | 0.6366 | 0.6152 | -0.0215 |
| 11 | 27 | 0.6831 | 0.7656 | +0.0826 |
| 12 | 46 | 0.5976 | 0.5795 | -0.0181 |

## Debug Images

- `outputs/experiments/seed2_failure_root_cause/debug_images/clean_detected_fixed_missed/01_2025-06-07-10-52-02ps5_NGyuan_tile_0012.jpg`
- `outputs/experiments/seed2_failure_root_cause/debug_images/clean_detected_fixed_missed/02_2025-05-16-16-19-14ps5_tile_0013.jpg`
- `outputs/experiments/seed2_failure_root_cause/debug_images/clean_detected_fixed_missed/03_2025-06-06-01-33-39ps5_NGyuan_tile_0003.jpg`
- `outputs/experiments/seed2_failure_root_cause/debug_images/clean_high_iou_fixed_worse/01_2025-06-07-10-52-02ps5_NGyuan_tile_0012.jpg`
- `outputs/experiments/seed2_failure_root_cause/debug_images/clean_high_iou_fixed_worse/02_2025-05-16-16-19-14ps5_tile_0013.jpg`
- `outputs/experiments/seed2_failure_root_cause/debug_images/clean_high_iou_fixed_worse/03_2025-06-06-01-33-39ps5_NGyuan_tile_0003.jpg`
- `outputs/experiments/seed2_failure_root_cause/debug_images/fixed_confidence_drop/01_2025-05-16-16-24-46ps5_tile_0002.jpg`
- `outputs/experiments/seed2_failure_root_cause/debug_images/fixed_confidence_drop/02_2025-05-25-12-13-55ps5_NGyuan_tile_0007.jpg`
- `outputs/experiments/seed2_failure_root_cause/debug_images/fixed_confidence_drop/03_2025-07-11-20-30-00ps5_NGyuan_tile_0011.jpg`
- `outputs/experiments/seed2_failure_root_cause/debug_images/fixed_new_high_conf_fp/01_2025-05-16-16-24-54ps5_tile_0012.jpg`
- `outputs/experiments/seed2_failure_root_cause/debug_images/fixed_new_high_conf_fp/02_2025-05-16-16-24-54ps5_tile_0007.jpg`
- `outputs/experiments/seed2_failure_root_cause/debug_images/fixed_new_high_conf_fp/03_2025-05-16-17-11-45ps5_tile_0006.jpg`

## NMS Note

Only post-NMS prediction JSON is available, so true pre-NMS ordering cannot be audited directly.
