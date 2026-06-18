# Ablation Table

| method | image augmentation | preserve original | weak attenuation | strict no-op | sampler_only | sampled distribution changed | pass count | mean delta vs clean P/R/mAP50/mAP50-95 | main conclusion |
|---|---|---|---|---|---|---|---|---|---|
| Clean YOLO default | false | false | false | false | false | false | reference | 0 / 0 / 0 / 0 | Baseline using YOLO default augmentation only. |
| fixed CATF-v2 | true | false | false | limited safety guards | false | false | 2/3 | +0.024700 / +0.001233 / +0.008800 / +0.008967 | Shows image augmentation can help, but seed2 fails due high-risk augmentation and non-active regression. |
| weak-only image augmentation | true | false | true | true | false | false | 2/3 | -0.018466 / +0.021777 / +0.004544 / +0.006586 | Repairs seed2 but damages seed0 because weak policy globally replaces safe fixed behavior. |
| preserve-weak image-only CATF | true | true | true | true | false | false | 3/3 | +0.026070 / +0.002821 / +0.012337 / +0.014467 | Preserves safe original policies for seed0/seed1 and applies weak/no-op control to repair seed2. Current image-only main-method candidate. |
| sampler_only demoted ablation note | false | n/a | n/a | n/a | true | true | not a main-method result | n/a | Excluded from the main method because it is weighted sampling or hard-example mining, not image data augmentation. |
