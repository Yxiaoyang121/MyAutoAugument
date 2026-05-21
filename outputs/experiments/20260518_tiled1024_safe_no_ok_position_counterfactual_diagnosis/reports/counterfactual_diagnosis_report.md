# Counterfactual Diagnosis Report

## Scope

- This run is prediction-only counterfactual diagnosis.
- It did not run YOLO train, final 50 epoch training, top3 short-training, or training dataset construction.
- Flow: baseline best.pt -> val FN detection -> counterfactual transforms on FN tiles -> re-predict -> recovery statistics.

## Direct Answers

1. Low-contrast/brightness transforms recovered `21` of `200` tested FN instances (`0.1050` unique recovery rate).
2. Highest recovery transform: `sharpen_mild` with recovery_rate `0.0700`.
3. CLAHE / contrast / gamma / brightness are counted as direct photometric support for `diag_policy_001`; per-op rates are listed below.
4. diag_policy_001 support: `supports diag_policy_001`.
5. copy_paste cannot be directly validated by this counterfactual because it changes training-set object frequency and placement; there is no new object inserted into the model weights during prediction-only inference.
6. Scale/size evidence is represented by `zoom_in_context`; if its recovery is materially higher than photometric transforms, that indicates a scale/context issue.
7. Photometric-recovered classes are listed under class recovery.
8. Classes with high unrecovered counts may need copy-paste, class-balanced augmentation, more samples, or annotation review.
9. Recommendation: `continue prioritizing diag_policy_001`.

## Existing Diagnosis Comparison

- Source diagnosis: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_policy_top3_from_baseline\diagnosis\diagnosis.json`
- Triggered issues: `low_contrast_missed_defect, low_contrast_missed_defect, class_imbalance, localization_bias`
- Contains `low_contrast_missed_defect`: `True`
- Counterfactual supports low-contrast issue: `True`

## Transform Recovery Table

| transform | tested FN | recovered | recovery_rate | low_contrast_rate | dark_rate | small_object_rate | main affected classes |
|---|---:|---:|---:|---:|---:|---:|---|
| `sharpen_mild` | 200 | 14 | 0.0700 | 0.0776 | 0.0745 | 0.0723 | 碰伤, 脏污, OK3, 油污, 轮廓划伤 |
| `contrast_up_130` | 200 | 12 | 0.0600 | 0.0517 | 0.0426 | 0.0542 | 碰伤, 脏污, OK3, 加强筋打伤, 轮廓划伤 |
| `contrast_up_115` | 200 | 10 | 0.0500 | 0.0431 | 0.0319 | 0.0482 | 碰伤, OK3, 脏污, 轮廓划伤, 锡尖 |
| `clahe_clip2` | 200 | 8 | 0.0400 | 0.0431 | 0.0426 | 0.0361 | 碰伤, OK3, 脏污, 油污 |
| `combined_photometric` | 200 | 8 | 0.0400 | 0.0345 | 0.0426 | 0.0361 | 碰伤, OK3, 脏污 |
| `clahe_clip3` | 200 | 6 | 0.0300 | 0.0345 | 0.0213 | 0.0301 | OK3, 碰伤, 脏污 |
| `gamma_brighten_085` | 200 | 5 | 0.0250 | 0.0172 | 0.0213 | 0.0181 | 碰伤, 漏背锡, 脏污 |
| `brightness_up_30` | 200 | 4 | 0.0200 | 0.0086 | 0.0000 | 0.0181 | 碰伤, 漏背锡 |
| `gamma_brighten_075` | 200 | 4 | 0.0200 | 0.0172 | 0.0213 | 0.0181 | 碰伤, 脏污 |
| `brightness_up_15` | 200 | 3 | 0.0150 | 0.0086 | 0.0000 | 0.0181 | 碰伤 |
| `zoom_in_context` | 200 | 2 | 0.0100 | 0.0086 | 0.0000 | 0.0060 | OK3, 脏污 |

## diag_policy_001 Operation Support

- `clahe`: recovered `10` / `200` unique FN, rate `0.0500`
- `contrast`: recovered `12` / `200` unique FN, rate `0.0600`
- `gamma`: recovered `5` / `200` unique FN, rate `0.0250`
- `brightness`: recovered `4` / `200` unique FN, rate `0.0200`
- `combined_photometric`: recovered `8` / `200` unique FN, rate `0.0400`

## Class Recovery

- Recovered: `碰伤` unique FN `9`
- Recovered: `脏污` unique FN `4`
- Recovered: `OK3` unique FN `3`
- Recovered: `加强筋打伤` unique FN `1`
- Recovered: `油污` unique FN `1`
- Recovered: `漏背锡` unique FN `1`
- Recovered: `轮廓划伤` unique FN `1`
- Recovered: `锡尖` unique FN `1`

## Still Unrecovered

- `碰伤` unrecovered `73` / `85` (`0.8588`)
- `轮廓划伤` unrecovered `31` / `33` (`0.9394`)
- `锡膏` unrecovered `17` / `17` (`1.0000`)
- `脏污` unrecovered `15` / `20` (`0.7500`)
- `油污` unrecovered `10` / `11` (`0.9091`)
- `漏背锡` unrecovered `9` / `10` (`0.9000`)
- `锡丝残留` unrecovered `7` / `7` (`1.0000`)
- `浅划伤` unrecovered `4` / `4` (`1.0000`)

## Counterfactual-Adjusted Policy Ranking

- Ranking starts from the existing proxy/safety ranking.
- Directly testable photometric and scale operations receive a recovery-rate adjustment.
- copy_paste is not hard rejected or directly penalized solely because prediction-only counterfactuals cannot validate new object synthesis.

| rank | policy_id | original_rank | base_score | recovery_score | adjusted_score | direct_ops | notes |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | `diag_policy_001` | 1 | 0.9570 | 0.0400 | 0.9650 | clahe, contrast, gamma, brightness | directly testable operations adjusted by FN recovery |
| 2 | `diag_policy_005` | 2 | 0.9561 | 0.0375 | 0.9636 | clahe, contrast, gamma, scale | copy_paste not directly testable by prediction-only counterfactual |
| 3 | `diag_policy_002` | 3 | 0.9536 | 0.0400 | 0.9616 | clahe, contrast, gamma, brightness | directly testable operations adjusted by FN recovery |
| 4 | `diag_policy_003` | 5 | 0.9283 | 0.0350 | 0.9353 | scale, contrast | copy_paste not directly testable by prediction-only counterfactual |
| 5 | `diag_policy_004` | 4 | 0.9305 | 0.0100 | 0.9325 | scale | directly testable operations adjusted by FN recovery |

Debug images: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_counterfactual_diagnosis\debug_images`
