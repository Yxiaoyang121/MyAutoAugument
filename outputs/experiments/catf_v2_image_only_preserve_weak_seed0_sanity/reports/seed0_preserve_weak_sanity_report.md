# Seed0 Preserve-Weak Image CATF Sanity Report

Scope: seed0-only 50ep sanity. No seed1/seed2, no multiseed, no sampler_only, no weighted index list.

## Decision Counts

- completed_50ep: `True`
- preserve_original: `9`
- weak_roi_texture: `0`
- strict_noop: `0`
- risk_levels: `{'low': 9}`

## Execution

- replay requested fixed classes: `[4, 11, 12]`
- executable policy classes observed: `[11]`
- ROI affected classes: `{'11': 20}`
- fixed class4/11/12 retained: `False`
- weak class9 replacement avoided: `True`
- image augmentation executed: `True`
- industrial images augmented: `20`
- ROI applied: `20`
- router random draw count: `1080`
- sampler_only_enabled: `False`
- sampler_only_effective: `False`
- weighted_index_list_enabled: `False`
- sampled_distribution_changed: `False`

## Metrics

| run | P | R | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| preserve-weak seed0 | 0.666585 | 0.730019 | 0.699934 | 0.466744 |
| clean seed0 baseline | 0.784600 | 0.676500 | 0.734700 | 0.475900 |
| fixed CATF-v2 seed0 | 0.778500 | 0.669700 | 0.743700 | 0.489500 |
| weak global seed0 | 0.679043 | 0.711821 | 0.722170 | 0.476005 |

Deltas:
- vs clean: `{'precision': -0.1180154143485671, 'recall': 0.05351944353035831, 'map50': -0.03476552076030226, 'map50_95': -0.009156493228598572}`
- vs fixed CATF-v2: `{'precision': -0.11191541434856711, 'recall': 0.06031944353035834, 'map50': -0.04376552076030227, 'map50_95': -0.022756493228598573}`
- vs weak global: `{'precision': -0.012458414348567093, 'recall': 0.018198443530358266, 'map50': -0.02223552076030222, 'map50_95': -0.009261493228598594}`

constraint_failed: `True`; reasons: `['precision_drop_gt_0.01', 'map50_drop_gt_0.01']`

## Interpretation

- The offline decision selection worked at the event level: all 9 feedback epochs selected `preserve_original`; weak and strict no-op were both 0.
- The run avoided the weak class9 replacement: class9 had no weak execution and no ROI applications.
- The run did not preserve the fixed seed0 class4/11/12 executable policy. The actual executable policy and ROI stats show only class11 augmentation.
- Therefore this sanity run failed. It does not validate the three-stage strategy yet.
- Do not proceed to seed2 until preserve_original execution installs or replays the fixed original class-op policy exactly and seed0 passes.
