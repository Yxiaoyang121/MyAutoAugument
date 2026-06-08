# Validation Error Diagnosis

- Status: completed
- TP: 723
- FP: 304
- FN: 168
- Precision: 0.7040
- Recall: 0.7989

## Diagnosis Vector
- small_object_score: 0.2496
- low_contrast_score: 0.5378
- class_imbalance_score: 0.9953
- localization_score: 0.0155
- false_positive_score: 0.2960

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=21 fn=0 precision=0.7529 recall=1.0000
- OK3: gt=274 tp=272 fp=67 fn=2 precision=0.8024 recall=0.9927
- 加强筋打伤: gt=8 tp=7 fp=2 fn=1 precision=0.7778 recall=0.8750
- 开裂: gt=2 tp=0 fp=7 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=7 fp=9 fn=11 precision=0.4375 recall=0.3889
- 浅划伤: gt=13 tp=9 fp=6 fn=3 precision=0.6000 recall=0.6923
- 漏背锡: gt=46 tp=32 fp=23 fn=11 precision=0.5818 recall=0.6957
- 碰伤: gt=269 tp=182 fp=71 fn=82 precision=0.7194 recall=0.6766
- 脏污: gt=42 tp=25 fp=27 fn=14 precision=0.4808 recall=0.5952
- 轮廓划伤: gt=84 tp=59 fp=33 fn=24 precision=0.6413 recall=0.7024
- 锡丝残留: gt=12 tp=9 fp=6 fn=3 precision=0.6000 recall=0.7500
- 锡尖: gt=27 tp=27 fp=15 fn=0 precision=0.6429 recall=1.0000
- 锡膏: gt=46 tp=30 fp=17 fn=15 precision=0.6383 recall=0.6522
