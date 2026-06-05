# Validation Error Diagnosis

- Status: completed
- TP: 712
- FP: 282
- FN: 174
- Precision: 0.7163
- Recall: 0.7867

## Diagnosis Vector
- small_object_score: 0.2598
- low_contrast_score: 0.5063
- class_imbalance_score: 0.9369
- localization_score: 0.0210
- false_positive_score: 0.2837

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=23 fn=0 precision=0.7356 recall=1.0000
- OK3: gt=274 tp=274 fp=70 fn=0 precision=0.7965 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=2 fn=1 precision=0.7778 recall=0.8750
- 开裂: gt=2 tp=1 fp=2 fn=1 precision=0.3333 recall=0.5000
- 油污: gt=18 tp=3 fp=14 fn=15 precision=0.1765 recall=0.1667
- 浅划伤: gt=13 tp=5 fp=0 fn=8 precision=1.0000 recall=0.3846
- 漏背锡: gt=46 tp=31 fp=17 fn=11 precision=0.6458 recall=0.6739
- 碰伤: gt=269 tp=200 fp=89 fn=63 precision=0.6920 recall=0.7435
- 脏污: gt=42 tp=20 fp=12 fn=20 precision=0.6250 recall=0.4762
- 轮廓划伤: gt=84 tp=46 fp=29 fn=32 precision=0.6133 recall=0.5476
- 锡丝残留: gt=12 tp=9 fp=5 fn=3 precision=0.6429 recall=0.7500
- 锡尖: gt=27 tp=22 fp=5 fn=5 precision=0.8148 recall=0.8148
- 锡膏: gt=46 tp=30 fp=14 fn=15 precision=0.6818 recall=0.6522
