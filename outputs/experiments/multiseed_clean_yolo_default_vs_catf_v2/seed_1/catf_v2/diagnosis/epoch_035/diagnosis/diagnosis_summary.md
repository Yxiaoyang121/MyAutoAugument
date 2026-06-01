# Validation Error Diagnosis

- Status: completed
- TP: 712
- FP: 310
- FN: 179
- Precision: 0.6967
- Recall: 0.7867

## Diagnosis Vector
- small_object_score: 0.2716
- low_contrast_score: 0.5291
- class_imbalance_score: 0.8786
- localization_score: 0.0155
- false_positive_score: 0.3033

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=20 fn=0 precision=0.7619 recall=1.0000
- OK3: gt=274 tp=272 fp=72 fn=1 precision=0.7907 recall=0.9927
- 加强筋打伤: gt=8 tp=8 fp=1 fn=0 precision=0.8889 recall=1.0000
- 开裂: gt=2 tp=2 fp=0 fn=0 precision=1.0000 recall=1.0000
- 油污: gt=18 tp=6 fp=22 fn=11 precision=0.2143 recall=0.3333
- 浅划伤: gt=13 tp=7 fp=16 fn=6 precision=0.3043 recall=0.5385
- 漏背锡: gt=46 tp=35 fp=30 fn=7 precision=0.5385 recall=0.7609
- 碰伤: gt=269 tp=180 fp=78 fn=84 precision=0.6977 recall=0.6691
- 脏污: gt=42 tp=20 fp=17 fn=20 precision=0.5405 recall=0.4762
- 轮廓划伤: gt=84 tp=57 fp=34 fn=26 precision=0.6264 recall=0.6786
- 锡丝残留: gt=12 tp=9 fp=9 fn=3 precision=0.5000 recall=0.7500
- 锡尖: gt=27 tp=24 fp=5 fn=3 precision=0.8276 recall=0.8889
- 锡膏: gt=46 tp=28 fp=6 fn=18 precision=0.8235 recall=0.6087
