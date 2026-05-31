# Validation Error Diagnosis

- Status: completed
- TP: 680
- FP: 297
- FN: 211
- Precision: 0.6960
- Recall: 0.7514

## Diagnosis Vector
- small_object_score: 0.3345
- low_contrast_score: 0.6130
- class_imbalance_score: 0.8980
- localization_score: 0.0155
- false_positive_score: 0.3040

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=20 fn=0 precision=0.7619 recall=1.0000
- OK3: gt=274 tp=273 fp=70 fn=1 precision=0.7959 recall=0.9964
- 加强筋打伤: gt=8 tp=7 fp=1 fn=1 precision=0.8750 recall=0.8750
- 开裂: gt=2 tp=2 fp=0 fn=0 precision=1.0000 recall=1.0000
- 油污: gt=18 tp=5 fp=15 fn=12 precision=0.2500 recall=0.2778
- 浅划伤: gt=13 tp=8 fp=27 fn=5 precision=0.2286 recall=0.6154
- 漏背锡: gt=46 tp=37 fp=40 fn=4 precision=0.4805 recall=0.8043
- 碰伤: gt=269 tp=152 fp=59 fn=114 precision=0.7204 recall=0.5651
- 脏污: gt=42 tp=19 fp=15 fn=22 precision=0.5588 recall=0.4524
- 轮廓划伤: gt=84 tp=49 fp=29 fn=31 precision=0.6282 recall=0.5833
- 锡丝残留: gt=12 tp=9 fp=7 fn=3 precision=0.5625 recall=0.7500
- 锡尖: gt=27 tp=27 fp=5 fn=0 precision=0.8438 recall=1.0000
- 锡膏: gt=46 tp=28 fp=9 fn=18 precision=0.7568 recall=0.6087
