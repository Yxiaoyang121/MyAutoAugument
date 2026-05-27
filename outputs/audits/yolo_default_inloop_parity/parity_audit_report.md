# YOLO Default In-Loop Parity Audit

## Runs

- Reference run: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_yolo_default_diagnosis_constrained_50ep\runs\yolo_default_seed42`
- In-loop no-feedback control run: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\yolo_default_inloop_no_feedback_control_50ep`

## Metrics

| metric | reference | control | control-reference |
|---|---:|---:|---:|
| Precision | 0.7132 | 0.7262 | 0.0130 |
| Recall | 0.7600 | 0.6844 | -0.0756 |
| mAP50 | 0.7759 | 0.7616 | -0.0142 |
| mAP50-95 | 0.5241 | 0.5250 | 0.0010 |

## Command Findings

- Reference uses `OnlineAugDetectionTrainer`: `true`
- Control declares default trainer: `true`
- Control uses custom trainer by command: `false`

Reference train command:

```text
YOLO.train model=yolo11n.pt data=E:\TJGY\MinPaper\MyAutoAugument\outputs\datasets\tiled\tiled_1024_ov20_full_safe_no_ok_position\data.yaml policy=E:\TJGY\MinPaper\MyAutoAugument\configs\online_policies\yolo_default_passthrough_policy.json epochs=50 imgsz=1024 batch=2 workers=0 device=0 seed=42 project=E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_yolo_default_diagnosis_constrained_50ep\runs\yolo_default_seed42 name=train trainer=OnlineAugDetectionTrainer
```

Control train command:

```text
YOLO.train model=yolo11n.pt data=outputs\datasets\tiled\tiled_1024_ov20_full_safe_no_ok_position\data.yaml epochs=50 imgsz=1024 batch=2 workers=0 device=0 seed=42 project=E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\yolo_default_inloop_no_feedback_control_50ep name=train UltralyticsDefaultDetectionTrainer yolo_default_augmentation_enabled=True disable_yolo_aug=False feedback_enabled=False industrial_aug_enabled=False
```

## args.yaml Diff

| key | reference | control |
|---|---|---|
| `data` | `E:\TJGY\MinPaper\MyAutoAugument\outputs\datasets\tiled\tiled_1024_ov20_full_safe_no_ok_position\data.yaml` | `outputs\datasets\tiled\tiled_1024_ov20_full_safe_no_ok_position\data.yaml` |
| `project` | `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_yolo_default_diagnosis_constrained_50ep\runs\yolo_default_seed42` | `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\yolo_default_inloop_no_feedback_control_50ep` |

## Augmentation Args Diff

No differences.

## Validation Args Diff

No differences.

## Learning-Rate Curve

- Rows compared: `50`
- Max abs LR delta: `0.000000000000`
- Last LR reference/control: `{'lr/pg0': '1.75224e-05', 'lr/pg1': '1.75224e-05', 'lr/pg2': '1.75224e-05'}` / `{'lr/pg0': '1.75224e-05', 'lr/pg1': '1.75224e-05', 'lr/pg2': '1.75224e-05'}`

## Results Curve

| metric | last reference | last control | last delta | max abs delta |
|---|---:|---:|---:|---:|
| Precision | 0.7356 | 0.6876 | -0.0480 | 0.1334 |
| Recall | 0.7330 | 0.7114 | -0.0216 | 0.1480 |
| mAP50 | 0.7666 | 0.7001 | -0.0666 | 0.0819 |
| mAP50-95 | 0.5220 | 0.4814 | -0.0407 | 0.0576 |

## close_mosaic

- Reference close_mosaic arg: `10`
- Control close_mosaic arg: `10`
- Reference log triggered close_mosaic: `true`
- Control log triggered close_mosaic: `true`

## Best.pt Selection

- Reference val uses best.pt: `true`
- Control val uses best.pt: `true`

## Industrial Augmentation

- Reference samples_augmented: `0`
- Control samples_augmented: `0`
- Reference ops: `{}`
- Control ops: `{}`

## Root Cause

- Custom entry changed behavior in completed control: `true`
- Primary: parity baseline mismatch: reference used custom passthrough OnlineAugDetectionTrainer, while the control used native trainer plus a no-op callback in the old script
- The selected reference run is not a pure native CLI run; it used OnlineAugDetectionTrainer with an empty passthrough policy.
- The completed in-loop control recorded epoch callback evidence, so the old no-feedback branch still attached a no-op callback.
- The reference used an absolute data YAML path while the control used the provided relative path.
- Project/save_dir differ by design and should not affect training math.

## Repair Recommendation

- Do not use the selected reference as proof of pure native YOLO default, because it used `OnlineAugDetectionTrainer` with a passthrough policy.
- For no-feedback parity, compare a native Python API smoke run against the in-loop no-feedback branch.
- The in-loop script has been repaired so `feedback=false` and `industrial_aug=false` directly calls native `YOLO.train(**same_args)` and registers no custom trainer, dataset, transform, or feedback callback.

## 1 Epoch Parity Smoke

- Native smoke run: `E:\TJGY\MinPaper\MyAutoAugument\outputs\audits\yolo_default_inloop_parity\smoke_native_yolo_default_1ep`
- In-loop no-feedback smoke run: `E:\TJGY\MinPaper\MyAutoAugument\outputs\audits\yolo_default_inloop_parity\smoke_inloop_no_feedback_1ep`
- Smoke completed: `true`
- args.yaml differences except project/save_dir: `0`
- metric/loss/lr max abs delta: `0.000000000000`
- in-loop epoch callback records: `0`
- smoke parity passed: `true`
