# Multiseed Clean YOLO Default vs In-Loop Feedback

## Setup

- Seeds: `0, 1, 2`
- Model/data: `yolo11n.pt`, `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml`
- Epochs/imgsz/batch/workers/device: `50 / 1024 / 2 / 0 / 0`
- Train images: `2301`
- Fixed augmented dataset generated: `false`
- Clean group: pure native Ultralytics `YOLO.train` with YOLO default augmentation enabled.
- Feedback group: single-run in-loop feedback, YOLO default augmentation enabled, industrial online augmentation enabled, copy_paste pending/not enabled.

## Per-Seed Results

| seed | group | Precision | Recall | mAP50 | mAP50-95 |
|---:|---|---:|---:|---:|---:|
| 0 | clean native | 0.7846 | 0.6765 | 0.7347 | 0.4759 |
| 0 | in-loop feedback | 0.7188 | 0.7273 | 0.7687 | 0.5276 |
| 1 | clean native | 0.7725 | 0.6477 | 0.7542 | 0.4799 |
| 1 | in-loop feedback | 0.7756 | 0.7276 | 0.7763 | 0.5147 |
| 2 | clean native | 0.6962 | 0.7286 | 0.7692 | 0.5224 |
| 2 | in-loop feedback | 0.6765 | 0.6741 | 0.7411 | 0.4993 |

## Per-Seed Deltas

| seed | Delta Precision | Delta Recall | Delta mAP50 | Delta mAP50-95 | constraint_failed | verdict |
|---:|---:|---:|---:|---:|---|---|
| 0 | -0.0658 | 0.0509 | 0.0340 | 0.0517 | `true` | mixed, constraint failed |
| 1 | 0.0031 | 0.0799 | 0.0221 | 0.0347 | `false` | feedback wins under constraints |
| 2 | -0.0197 | -0.0546 | -0.0281 | -0.0231 | `true` | feedback regressed |

## Mean +/- Std

Sample standard deviation across seeds `0/1/2`.

| metric | clean native | in-loop feedback | delta |
|---|---:|---:|---:|
| Precision | 0.7511 +/- 0.0479 | 0.7236 +/- 0.0497 | -0.0275 +/- 0.0351 |
| Recall | 0.6843 +/- 0.0410 | 0.7097 +/- 0.0308 | 0.0254 +/- 0.0708 |
| mAP50 | 0.7527 +/- 0.0173 | 0.7620 +/- 0.0185 | 0.0093 +/- 0.0330 |
| mAP50-95 | 0.4927 +/- 0.0258 | 0.5138 +/- 0.0142 | 0.0211 +/- 0.0392 |

## Win Counts

- Feedback wins under industrial constraints: `1/3`
- Constraint failed seeds: `2/3`
- All-metric win seeds: `1/3`
- Precision wins: `1/3`
- Recall wins: `2/3`
- mAP50 wins: `2/3`
- mAP50-95 wins: `2/3`

## Harm Cases

- Seed `0`: Precision -0.0658
- Seed `2`: Precision -0.0197, mAP50 -0.0281, mAP50-95 -0.0231, Recall -0.0546

## Policy History Summary

- Most frequently upregulated ops: `clahe` (104), `gamma` (104), `sharpen_mild` (95), `local_contrast` (95), `cutout_safe` (54), `brightness` (50)
- Most frequently downregulated ops: `clahe` (54), `gamma` (54), `brightness` (50), `cutout_safe` (46)
- Update epochs per feedback run: `5, 10, 15, 20, 25, 30, 35, 40, 45`.

## Conclusion

- Recommendation: `not_main_result_yet`
- The feedback controller is promising but not stable enough to be the paper main result as-is.
- Seed 1 is a clean win, seed 0 improves Recall/mAP but violates Precision, and seed 2 regresses across all four metrics.
- Use this as an ablation/diagnostic result unless the controller is made more conservative and passes another multiseed check.
