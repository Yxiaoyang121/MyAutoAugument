# Operator Impact

Impact is measured at policy-group scope because this experiment does not run single-operator ablations.

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
