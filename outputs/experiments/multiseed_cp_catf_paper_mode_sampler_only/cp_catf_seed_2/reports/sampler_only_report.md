# CP-CATF Paper-Mode Sampler-Only Seed 2

## Result

- run_dir: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\multiseed_cp_catf_paper_mode_sampler_only\cp_catf_seed_2`
- paper_mode: `true`
- final_val_leakage_false: `true`
- final_val_used_for_policy_selection: `false`
- constraint_failed: `false`
- constraint_reasons: `none`

| run | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| clean paper seed2 | 0.6290 | 0.6385 | 0.6590 | 0.4381 |
| CP-CATF sampler-only seed2 | 0.7062 | 0.6512 | 0.6843 | 0.4427 |
| delta | +0.0772 | +0.0127 | +0.0253 | +0.0046 |

## Sampler-Only Status

- sampler_only_effective: `true`
- effective feedback epochs: `[5, 10, 15, 20, 25, 30, 35, 40, 45]`
- weighted_train_core_images_count: `176`
- weighted_sampler_enabled: `false`
- weighted_index_list_enabled: `true`
- sampled_distribution_changed: `true`

## Augmentation Counts

- image_augmented: `0`
- ROI applied: `0`
- industrial image augmented: `0`
- router random draw count: `0`

## Weighted Classes

| class_id | class_name | dominant_issue | weight | reasons |
|---:|---|---|---:|---|
| 6 | 漏背锡 | low_contrast_fn | 1.2500 | low_contrast_fn |
| 9 | 轮廓划伤 | texture_boundary_weak | 1.2500 | texture_boundary_weak |

## Protected-Class Check

- high-FP / OK / no_aug wrongly weighted: `false`
- protected weighted count: `0`

## Feedback Decisions

| epoch | candidate_policy_id | action | class | score | image rejected | sampler effective | weighted images | distribution changed |
|---:|---|---|---:|---:|---|---|---:|---|
| 5 | candidate_policy_3_sampler_only | sampler_only | 8 | -0.1520 | true | true | 123 | true |
| 10 | candidate_policy_3_sampler_only | sampler_only | 8 | -0.1245 | true | true | 123 | true |
| 15 | candidate_policy_3_sampler_only | sampler_only | -1 | 0.0000 | true | true | 29 | true |
| 20 | candidate_policy_3_sampler_only | sampler_only | 12 | -0.1000 | true | true | 26 | true |
| 25 | candidate_policy_3_sampler_only | sampler_only | -1 | 0.0000 | true | true | 44 | true |
| 30 | candidate_policy_3_sampler_only | sampler_only | 6 | -0.1208 | true | true | 50 | true |
| 35 | candidate_policy_3_sampler_only | sampler_only | 9 | -0.0974 | true | true | 123 | true |
| 40 | candidate_policy_3_sampler_only | sampler_only | -1 | 0.0000 | true | true | 152 | true |
| 45 | candidate_policy_3_sampler_only | sampler_only | 6 | -0.0859 | true | true | 176 | true |

## Required Artifacts

- final_metrics: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\multiseed_cp_catf_paper_mode_sampler_only\cp_catf_seed_2\reports\final_metrics.json`
- sampler_only_report_md: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\multiseed_cp_catf_paper_mode_sampler_only\cp_catf_seed_2\reports\sampler_only_report.md`
- sampler_only_report_json: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\multiseed_cp_catf_paper_mode_sampler_only\cp_catf_seed_2\reports\sampler_only_report.json`
- sample_weight_map: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\multiseed_cp_catf_paper_mode_sampler_only\cp_catf_seed_2\reports\sample_weight_map.json`
- weighted_train_indices: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\multiseed_cp_catf_paper_mode_sampler_only\cp_catf_seed_2\reports\weighted_train_indices.json`
- sampled_distribution_before_after: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\multiseed_cp_catf_paper_mode_sampler_only\cp_catf_seed_2\reports\sampled_distribution_before_after.json`
- policy_history: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\multiseed_cp_catf_paper_mode_sampler_only\cp_catf_seed_2\reports\policy_history.json`
- class_policy_history: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\multiseed_cp_catf_paper_mode_sampler_only\cp_catf_seed_2\reports\class_policy_history.json`
- causal_probe_decisions_used: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\multiseed_cp_catf_paper_mode_sampler_only\cp_catf_seed_2\reports\causal_probe_decisions_used.json`
- online_aug_stats: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\multiseed_cp_catf_paper_mode_sampler_only\cp_catf_seed_2\reports\online_aug_stats.json`
- roi_aug_stats: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\multiseed_cp_catf_paper_mode_sampler_only\cp_catf_seed_2\reports\roi_aug_stats.json`
