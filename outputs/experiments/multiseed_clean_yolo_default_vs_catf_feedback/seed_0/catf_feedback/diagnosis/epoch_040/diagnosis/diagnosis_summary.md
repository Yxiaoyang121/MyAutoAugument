# Validation Error Diagnosis

- Status: completed
- TP: 708
- FP: 278
- FN: 178
- Precision: 0.7181
- Recall: 0.7823

## Diagnosis Vector
- small_object_score: 0.2750
- low_contrast_score: 0.5183
- class_imbalance_score: 0.9953
- localization_score: 0.0210
- false_positive_score: 0.2819

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=24 fn=0 precision=0.7273 recall=1.0000
- OK3: gt=274 tp=273 fp=71 fn=1 precision=0.7936 recall=0.9964
- 加强筋打伤: gt=8 tp=7 fp=0 fn=1 precision=1.0000 recall=0.8750
- 开裂: gt=2 tp=0 fp=7 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=7 fp=15 fn=10 precision=0.3182 recall=0.3889
- 浅划伤: gt=13 tp=8 fp=5 fn=4 precision=0.6154 recall=0.6154
- 漏背锡: gt=46 tp=29 fp=25 fn=13 precision=0.5370 recall=0.6304
- 碰伤: gt=269 tp=183 fp=58 fn=81 precision=0.7593 recall=0.6803
- 脏污: gt=42 tp=20 fp=22 fn=18 precision=0.4762 recall=0.4762
- 轮廓划伤: gt=84 tp=55 fp=28 fn=25 precision=0.6627 recall=0.6548
- 锡丝残留: gt=12 tp=9 fp=7 fn=3 precision=0.5625 recall=0.7500
- 锡尖: gt=27 tp=25 fp=2 fn=2 precision=0.9259 recall=0.9259
- 锡膏: gt=46 tp=28 fp=14 fn=18 precision=0.6667 recall=0.6087
