# Random External Policy Proxy/Safety Report

- Scope: pre-training proxy/safety audit for one randomly sampled external augmentation policy.
- hard_filter_pass: `true`
- proxy_score: `0.958779`
- safety_score: `0.940000`
- combined_proxy_safety_score: `0.950328`
- bbox_valid_rate: `1.000000`
- original_bbox_retention: `1.000000`
- class_id_valid_rate: `1.000000`
- exposure_score: `0.882225`
- image_save_failures: `0`
- label_valid_failures: `0`

## Operation Applications

| operation | applied images |
|---|---:|
| brightness | 1581 |
| cutout | 370 |
| horizontal_flip | 1036 |
| sharpen | 561 |

## Safety Soft Penalties

- None.
