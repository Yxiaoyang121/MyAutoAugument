# Validation Error Diagnosis

- Status: completed
- TP: 702
- FP: 276
- FN: 191
- Precision: 0.7178
- Recall: 0.7757

## Diagnosis Vector
- small_object_score: 0.3022
- low_contrast_score: 0.5801
- class_imbalance_score: 0.9369
- localization_score: 0.0133
- false_positive_score: 0.2822

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=20 fn=0 precision=0.7619 recall=1.0000
- OK3: gt=274 tp=274 fp=69 fn=0 precision=0.7988 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=0 fn=1 precision=1.0000 recall=0.8750
- 开裂: gt=2 tp=2 fp=4 fn=0 precision=0.3333 recall=1.0000
- 油污: gt=18 tp=3 fp=10 fn=15 precision=0.2308 recall=0.1667
- 浅划伤: gt=13 tp=7 fp=15 fn=5 precision=0.3182 recall=0.5385
- 漏背锡: gt=46 tp=37 fp=21 fn=8 precision=0.6379 recall=0.8043
- 碰伤: gt=269 tp=170 fp=57 fn=92 precision=0.7489 recall=0.6320
- 脏污: gt=42 tp=24 fp=16 fn=17 precision=0.6000 recall=0.5714
- 轮廓划伤: gt=84 tp=46 fp=38 fn=36 precision=0.5476 recall=0.5476
- 锡丝残留: gt=12 tp=10 fp=7 fn=2 precision=0.5882 recall=0.8333
- 锡尖: gt=27 tp=27 fp=7 fn=0 precision=0.7941 recall=1.0000
- 锡膏: gt=46 tp=31 fp=12 fn=15 precision=0.7209 recall=0.6739
