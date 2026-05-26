# YOLO Default Diagnosis-Constrained Experiment

- Run ID: `20260518_tiled1024_safe_no_ok_position_yolo_default_diagnosis_constrained_50ep`
- Base policy: Ultralytics YOLO default augmentation remains enabled for A-D.
- Custom diagnosis policies are added online in the dataloader; no fixed augmented dataset is generated.
- Constraint reference: `YOLO default`
- Constraint: Precision delta >= `-0.02`, mAP50 delta >= `-0.01`, mAP50-95 delta >= `-0.01`.

## Unified Metrics

| run | Precision | Recall | mAP50 | mAP50-95 | dP | dR | d_mAP50 | d_mAP50-95 | pass constraints |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| YOLO default | 0.7132 | 0.7600 | 0.7759 | 0.5241 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | yes |
| YOLO default + diagnosis_light | 0.7343 | 0.6862 | 0.7555 | 0.4993 | +0.0211 | -0.0738 | -0.0204 | -0.0247 | no |
| YOLO default + diagnosis_precision_safe | 0.6647 | 0.7493 | 0.7557 | 0.5055 | -0.0485 | -0.0107 | -0.0202 | -0.0185 | no |
| YOLO default + diagnosis_recall_safe | 0.6712 | 0.7659 | 0.7411 | 0.5044 | -0.0420 | +0.0059 | -0.0348 | -0.0196 | no |
| custom_yolo_like_base old control | 0.6661 | 0.7490 | 0.7129 | 0.4415 | -0.0471 | -0.0110 | -0.0629 | -0.0826 | no |

## Constraint Selection

- Best under constraints: `YOLO default`
- Recall improved under constraints: `false`
- Passed candidates: `yolo_default`
- Failure driver if no improvement: `diagnosis_light: mAP50-95 dropped without localization_weak increase in conf=0.25 diagnosis; diagnosis_precision_safe: FP increased by 18; diagnosis_precision_safe: mAP50-95 dropped without localization_weak increase in conf=0.25 diagnosis; diagnosis_recall_safe: mAP50-95 dropped without localization_weak increase in conf=0.25 diagnosis`

## Diagnostics

| run | TP | FP | FN | localization_weak | top FN class | top FP class |
|---|---:|---:|---:|---:|---|---|
| YOLO default | 719 | 287 | 162 | 24 | 碰伤 (fn=79) | 碰伤 (fp=78) |
| YOLO default + diagnosis_light | 720 | 278 | 172 | 13 | 碰伤 (fn=75) | 碰伤 (fp=75) |
| YOLO default + diagnosis_precision_safe | 725 | 305 | 156 | 24 | 碰伤 (fn=75) | 碰伤 (fp=76) |
| YOLO default + diagnosis_recall_safe | 735 | 269 | 149 | 21 | 碰伤 (fn=72) | 碰伤 (fp=72) |
| custom_yolo_like_base old control | 722 | 319 | 170 | 13 | 碰伤 (fn=71) | 碰伤 (fp=107) |

## Operator Impact

Operator impact is reported at policy-group scope, not as isolated single-op ablation.

| group | op | prob | strength | applied | dP | dR | d_mAP50 | d_mAP50-95 | scope |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| yolo_default | ultralytics_default_aug_pool | n/a | n/a | None | +0.0000 | +0.0000 | +0.0000 | +0.0000 | reference_default_pool |
| diagnosis_light | hsv_jitter | 0.1500 | 0.2500 | 17396 | +0.0211 | -0.0738 | -0.0204 | -0.0247 | policy_group_not_isolated |
| diagnosis_light | random_scale_translate | 0.1000 | 0.2000 | 11278 | +0.0211 | -0.0738 | -0.0204 | -0.0247 | policy_group_not_isolated |
| diagnosis_precision_safe | hsv_jitter | 0.1000 | 0.2000 | 11504 | -0.0485 | -0.0107 | -0.0202 | -0.0185 | policy_group_not_isolated |
| diagnosis_precision_safe | random_scale_translate | 0.0800 | 0.1500 | 9160 | -0.0485 | -0.0107 | -0.0202 | -0.0185 | policy_group_not_isolated |
| diagnosis_precision_safe | mosaic4 | 0.0300 | 0.2500 | 2707 | -0.0485 | -0.0107 | -0.0202 | -0.0185 | policy_group_not_isolated |
| diagnosis_precision_safe | random_erasing | 0.0300 | 0.1200 | 3434 | -0.0485 | -0.0107 | -0.0202 | -0.0185 | policy_group_not_isolated |
| diagnosis_recall_safe | clahe | 0.1200 | 0.2200 | 13804 | -0.0420 | +0.0059 | -0.0348 | -0.0196 | policy_group_not_isolated |
| diagnosis_recall_safe | contrast | 0.1200 | 0.2000 | 13840 | -0.0420 | +0.0059 | -0.0348 | -0.0196 | policy_group_not_isolated |
| diagnosis_recall_safe | gamma | 0.1000 | 0.1800 | 11483 | -0.0420 | +0.0059 | -0.0348 | -0.0196 | policy_group_not_isolated |
| diagnosis_recall_safe | brightness | 0.0800 | 0.1400 | 9112 | -0.0420 | +0.0059 | -0.0348 | -0.0196 | policy_group_not_isolated |
| diagnosis_recall_safe | sharpen_mild | 0.1000 | 0.2000 | 11533 | -0.0420 | +0.0059 | -0.0348 | -0.0196 | policy_group_not_isolated |
| diagnosis_recall_safe | random_scale_translate | 0.1200 | 0.2000 | 13662 | -0.0420 | +0.0059 | -0.0348 | -0.0196 | policy_group_not_isolated |
| custom_yolo_like_base_old | mosaic4 | 1.0000 | 1.0000 | 92040 | -0.0471 | -0.0110 | -0.0629 | -0.0826 | policy_group_not_isolated |
| custom_yolo_like_base_old | hsv_jitter | 1.0000 | 1.0000 | 115050 | -0.0471 | -0.0110 | -0.0629 | -0.0826 | policy_group_not_isolated |
| custom_yolo_like_base_old | random_scale_translate | 1.0000 | 1.0000 | 115050 | -0.0471 | -0.0110 | -0.0629 | -0.0826 | policy_group_not_isolated |
| custom_yolo_like_base_old | horizontal_flip | 0.5000 | 1.0000 | 57654 | -0.0471 | -0.0110 | -0.0629 | -0.0826 | policy_group_not_isolated |
| custom_yolo_like_base_old | randaugment_like | 0.4500 | 0.4500 | 51325 | -0.0471 | -0.0110 | -0.0629 | -0.0826 | policy_group_not_isolated |
| custom_yolo_like_base_old | random_erasing | 0.4000 | 0.6000 | 46039 | -0.0471 | -0.0110 | -0.0629 | -0.0826 | policy_group_not_isolated |

## Artifacts

- Metrics JSON: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_yolo_default_diagnosis_constrained_50ep\reports\diagnosis_constrained_metrics.json`
- Constraint scoring JSON: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_yolo_default_diagnosis_constrained_50ep\reports\constraint_scoring.json`
- Operator impact JSON: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_yolo_default_diagnosis_constrained_50ep\reports\operator_impact.json`
