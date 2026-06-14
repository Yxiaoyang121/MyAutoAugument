# Validation Error Diagnosis

- Status: completed
- TP: 628
- FP: 545
- FN: 245
- Precision: 0.5354
- Recall: 0.6939

## Diagnosis Vector
- small_object_score: 0.3616
- low_contrast_score: 0.4696
- class_imbalance_score: 0.9953
- localization_score: 0.0354
- false_positive_score: 0.4646

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=12 fn=0 precision=0.8421 recall=1.0000
- OK3: gt=274 tp=270 fp=66 fn=4 precision=0.8036 recall=0.9854
- 加强筋打伤: gt=8 tp=0 fp=0 fn=8 precision=0.0000 recall=0.0000
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=0 fp=0 fn=18 precision=0.0000 recall=0.0000
- 浅划伤: gt=13 tp=1 fp=0 fn=12 precision=1.0000 recall=0.0769
- 漏背锡: gt=46 tp=24 fp=77 fn=15 precision=0.2376 recall=0.5217
- 碰伤: gt=269 tp=178 fp=170 fn=79 precision=0.5115 recall=0.6617
- 脏污: gt=42 tp=13 fp=103 fn=29 precision=0.1121 recall=0.3095
- 轮廓划伤: gt=84 tp=40 fp=65 fn=35 precision=0.3810 recall=0.4762
- 锡丝残留: gt=12 tp=0 fp=0 fn=12 precision=0.0000 recall=0.0000
- 锡尖: gt=27 tp=8 fp=1 fn=19 precision=0.8889 recall=0.2963
- 锡膏: gt=46 tp=30 fp=51 fn=12 precision=0.3704 recall=0.6522
