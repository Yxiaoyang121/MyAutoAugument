# Validation Error Diagnosis

- Status: completed
- TP: 723
- FP: 265
- FN: 167
- Precision: 0.7318
- Recall: 0.7989

## Diagnosis Vector
- small_object_score: 0.2479
- low_contrast_score: 0.5527
- class_imbalance_score: 0.9369
- localization_score: 0.0166
- false_positive_score: 0.2682

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=19 fn=0 precision=0.7711 recall=1.0000
- OK3: gt=274 tp=274 fp=66 fn=0 precision=0.8059 recall=1.0000
- 加强筋打伤: gt=8 tp=6 fp=2 fn=2 precision=0.7500 recall=0.7500
- 开裂: gt=2 tp=2 fp=3 fn=0 precision=0.4000 recall=1.0000
- 油污: gt=18 tp=3 fp=21 fn=12 precision=0.1250 recall=0.1667
- 浅划伤: gt=13 tp=8 fp=6 fn=5 precision=0.5714 recall=0.6154
- 漏背锡: gt=46 tp=33 fp=14 fn=8 precision=0.7021 recall=0.7174
- 碰伤: gt=269 tp=189 fp=81 fn=77 precision=0.7000 recall=0.7026
- 脏污: gt=42 tp=26 fp=14 fn=15 precision=0.6500 recall=0.6190
- 轮廓划伤: gt=84 tp=54 fp=19 fn=27 precision=0.7397 recall=0.6429
- 锡丝残留: gt=12 tp=9 fp=6 fn=3 precision=0.6000 recall=0.7500
- 锡尖: gt=27 tp=26 fp=2 fn=1 precision=0.9286 recall=0.9630
- 锡膏: gt=46 tp=29 fp=12 fn=17 precision=0.7073 recall=0.6304
