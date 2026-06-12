# CP-CATF Paper-Mode Sampler-Only Seed 0

## Result

- run_dir: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\cp_catf_paper_mode_sampler_only_seed0`
- paper_mode: `true`
- final_val_leakage_false: `true`
- final_val_used_for_policy_selection: `false`
- constraint_failed: `false`
- constraint_reasons: `none`

| run | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| clean paper seed0 | 0.7513 | 0.6763 | 0.7566 | 0.5114 |
| CP-CATF sampler-only seed0 | 0.7588 | 0.6878 | 0.7779 | 0.5204 |
| delta | +0.0075 | +0.0115 | +0.0213 | +0.0090 |

## Sampler-Only Status

- sampler_only_effective: `true`
- effective feedback epochs: `[15, 20, 25, 30, 35, 40, 45]`
- weighted_train_core_images_count: `72`
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
| 2 | 加强筋打伤 | low_support | 1.2000 | low_support |
| 10 | 锡丝残留 | low_support | 1.2000 | low_support |
| 11 | 锡尖 | texture_boundary_weak | 1.2500 | texture_boundary_weak |
| 12 | 锡膏 | texture_boundary_weak | 1.2500 | texture_boundary_weak |

## Protected-Class Check

- high-FP / OK / no_aug wrongly weighted: `false`
- protected weighted count: `0`

## Feedback Decisions

| epoch | candidate_policy_id | action | class | score | image rejected | sampler effective | weighted images | distribution changed |
|---:|---|---|---:|---:|---|---|---:|---|
| 5 | candidate_policy_3_sampler_only | sampler_only | 8 | -0.1520 | true | false | 0 | false |
| 10 | candidate_policy_3_sampler_only | sampler_only | 8 | -0.1005 | true | false | 0 | false |
| 15 | candidate_policy_3_sampler_only | sampler_only | 12 | -0.1245 | true | true | 34 | true |
| 20 | candidate_policy_3_sampler_only | sampler_only | 8 | -0.0832 | true | true | 55 | true |
| 25 | candidate_policy_3_sampler_only | sampler_only | 6 | -0.1110 | true | true | 50 | true |
| 30 | candidate_policy_3_sampler_only | sampler_only | 6 | -0.0986 | true | true | 50 | true |
| 35 | candidate_policy_3_sampler_only | sampler_only | 9 | -0.1167 | true | true | 123 | true |
| 40 | candidate_policy_3_sampler_only | sampler_only | -1 | 0.0000 | true | true | 13 | true |
| 45 | candidate_policy_3_sampler_only | sampler_only | -1 | 0.0000 | true | true | 72 | true |

## Required Artifacts

- final_metrics: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\cp_catf_paper_mode_sampler_only_seed0\reports\final_metrics.json`
- sampler_only_report_md: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\cp_catf_paper_mode_sampler_only_seed0\reports\sampler_only_report.md`
- sampler_only_report_json: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\cp_catf_paper_mode_sampler_only_seed0\reports\sampler_only_report.json`
- sample_weight_map: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\cp_catf_paper_mode_sampler_only_seed0\reports\sample_weight_map.json`
- weighted_train_indices: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\cp_catf_paper_mode_sampler_only_seed0\reports\weighted_train_indices.json`
- sampled_distribution_before_after: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\cp_catf_paper_mode_sampler_only_seed0\reports\sampled_distribution_before_after.json`
- policy_history: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\cp_catf_paper_mode_sampler_only_seed0\reports\policy_history.json`
- class_policy_history: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\cp_catf_paper_mode_sampler_only_seed0\reports\class_policy_history.json`
- causal_probe_decisions_used: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\cp_catf_paper_mode_sampler_only_seed0\reports\causal_probe_decisions_used.json`
- online_aug_stats: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\cp_catf_paper_mode_sampler_only_seed0\reports\online_aug_stats.json`
- roi_aug_stats: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\cp_catf_paper_mode_sampler_only_seed0\reports\roi_aug_stats.json`
