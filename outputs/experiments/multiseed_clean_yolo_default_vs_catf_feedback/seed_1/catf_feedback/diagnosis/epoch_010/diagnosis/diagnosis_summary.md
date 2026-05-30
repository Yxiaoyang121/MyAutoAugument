# Validation Error Diagnosis

- Status: completed
- TP: 595
- FP: 351
- FN: 286
- Precision: 0.6290
- Recall: 0.6575

## Diagnosis Vector
- small_object_score: 0.4075
- low_contrast_score: 0.5016
- class_imbalance_score: 0.9953
- localization_score: 0.0265
- false_positive_score: 0.3710

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=42 fn=0 precision=0.6038 recall=1.0000
- OK3: gt=274 tp=265 fp=60 fn=9 precision=0.8154 recall=0.9672
- 加强筋打伤: gt=8 tp=0 fp=0 fn=8 precision=0.0000 recall=0.0000
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=4 fp=36 fn=14 precision=0.1000 recall=0.2222
- 浅划伤: gt=13 tp=0 fp=0 fn=13 precision=0.0000 recall=0.0000
- 漏背锡: gt=46 tp=17 fp=13 fn=26 precision=0.5667 recall=0.3696
- 碰伤: gt=269 tp=167 fp=105 fn=88 precision=0.6140 recall=0.6208
- 脏污: gt=42 tp=4 fp=0 fn=36 precision=1.0000 recall=0.0952
- 轮廓划伤: gt=84 tp=20 fp=19 fn=61 precision=0.5128 recall=0.2381
- 锡丝残留: gt=12 tp=0 fp=0 fn=12 precision=0.0000 recall=0.0000
- 锡尖: gt=27 tp=25 fp=22 fn=2 precision=0.5319 recall=0.9259
- 锡膏: gt=46 tp=29 fp=54 fn=15 precision=0.3494 recall=0.6304
