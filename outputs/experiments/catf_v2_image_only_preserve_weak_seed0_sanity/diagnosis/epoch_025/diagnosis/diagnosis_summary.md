# Validation Error Diagnosis

- Status: completed
- TP: 730
- FP: 369
- FN: 153
- Precision: 0.6642
- Recall: 0.8066

## Diagnosis Vector
- small_object_score: 0.2445
- low_contrast_score: 0.5490
- class_imbalance_score: 0.9953
- localization_score: 0.0243
- false_positive_score: 0.3358

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=22 fn=0 precision=0.7442 recall=1.0000
- OK3: gt=274 tp=273 fp=66 fn=1 precision=0.8053 recall=0.9964
- 加强筋打伤: gt=8 tp=7 fp=1 fn=1 precision=0.8750 recall=0.8750
- 开裂: gt=2 tp=0 fp=1 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=6 fp=12 fn=12 precision=0.3333 recall=0.3333
- 浅划伤: gt=13 tp=7 fp=2 fn=6 precision=0.7778 recall=0.5385
- 漏背锡: gt=46 tp=37 fp=55 fn=4 precision=0.4022 recall=0.8043
- 碰伤: gt=269 tp=200 fp=111 fn=63 precision=0.6431 recall=0.7435
- 脏污: gt=42 tp=20 fp=17 fn=18 precision=0.5405 recall=0.4762
- 轮廓划伤: gt=84 tp=46 fp=40 fn=33 precision=0.5349 recall=0.5476
- 锡丝残留: gt=12 tp=9 fp=8 fn=3 precision=0.5294 recall=0.7500
- 锡尖: gt=27 tp=25 fp=7 fn=2 precision=0.7812 recall=0.9259
- 锡膏: gt=46 tp=36 fp=27 fn=8 precision=0.5714 recall=0.7826
