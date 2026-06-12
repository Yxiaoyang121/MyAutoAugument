# CP-CATF Paper-Mode Sampler-Only Multiseed Summary

## Conclusion

- 3/3 pass: `false`
- pass_count: `2/3`
- paper main-method candidate: `false`
- image augmentation zero all seeds: `true`
- final val leakage false all seeds: `true`
- sampled distribution changed all seeds: `true`

Seed1 fails the requested constraints, so this multiseed run should not be reported as the paper main method.

## Metrics

| seed | clean P | clean R | clean mAP50 | clean mAP50-95 | CP P | CP R | CP mAP50 | CP mAP50-95 | dP | dR | dM50 | dM95 | constraint_failed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.7513 | 0.6763 | 0.7566 | 0.5114 | 0.7588 | 0.6878 | 0.7779 | 0.5204 | +0.0075 | +0.0115 | +0.0213 | +0.0090 | false |
| 1 | 0.7220 | 0.7582 | 0.7777 | 0.5251 | 0.7085 | 0.7258 | 0.7412 | 0.5112 | -0.0135 | -0.0324 | -0.0365 | -0.0139 | true |
| 2 | 0.6290 | 0.6385 | 0.6590 | 0.4381 | 0.7062 | 0.6512 | 0.6843 | 0.4427 | +0.0772 | +0.0127 | +0.0253 | +0.0046 | false |
| mean | 0.7008 | 0.6910 | 0.7311 | 0.4915 | 0.7245 | 0.6883 | 0.7345 | 0.4914 | +0.0237 | -0.0027 | +0.0034 | -0.0001 | n/a |

## Sampler-Only Coverage

- sampler_only effective seed count: `3/3`
- sampler_only effective feedback event count: `25`
- weighted train_core images total: `371`
- weighted train_core images by seed: `{'0': 72, '1': 123, '2': 176}`

| seed | effective epochs | weighted images | weighted_index_list_enabled | sampled distribution changed | image augmented | ROI applied | router draws | final val leakage false |
|---:|---|---:|---|---|---:|---:|---:|---|
| 0 | [15, 20, 25, 30, 35, 40, 45] | 72 | true | true | 0 | 0 | 0 | true |
| 1 | [5, 10, 15, 20, 25, 30, 35, 40, 45] | 123 | true | true | 0 | 0 | 0 | true |
| 2 | [5, 10, 15, 20, 25, 30, 35, 40, 45] | 176 | true | true | 0 | 0 | 0 | true |

## Interpretation

- Development-mode difference: Development-mode CP-CATF used existing validation diagnostics and included executable image/ROI augmentation evidence, so it is not the leakage-free paper-mode result.
- No-op paper-mode difference: The no-op paper-mode run had sampler-only pending or ineffective and did not change train sampling; this sampler-only run changes the dataloader via weighted index lists.
- Weak image augmentation / attenuation: Do not promote weak image augmentation as the current main result. With seed1 failing constraints, first analyze sampler-only failure modes; weak_image_aug or attenuation remains an extension direction.

## Report Paths

- seed0: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\cp_catf_paper_mode_sampler_only_seed0\reports\sampler_only_report.md`
- seed1: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\multiseed_cp_catf_paper_mode_sampler_only\cp_catf_seed_1\reports\sampler_only_report.md`
- seed2: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\multiseed_cp_catf_paper_mode_sampler_only\cp_catf_seed_2\reports\sampler_only_report.md`
- multiseed JSON: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\multiseed_cp_catf_paper_mode_sampler_only\reports\multiseed_sampler_only_summary.json`
- multiseed Markdown: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\multiseed_cp_catf_paper_mode_sampler_only\reports\multiseed_sampler_only_summary.md`
