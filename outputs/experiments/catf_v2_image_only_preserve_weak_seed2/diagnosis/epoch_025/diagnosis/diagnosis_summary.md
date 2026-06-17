# Validation Error Diagnosis

- Status: completed
- TP: 728
- FP: 383
- FN: 156
- Precision: 0.6553
- Recall: 0.8044

## Diagnosis Vector
- small_object_score: 0.2581
- low_contrast_score: 0.5247
- class_imbalance_score: 0.9175
- localization_score: 0.0232
- false_positive_score: 0.3447

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=24 fn=0 precision=0.7273 recall=1.0000
- OK3: gt=274 tp=274 fp=65 fn=0 precision=0.8083 recall=1.0000
- 加强筋打伤: gt=8 tp=3 fp=0 fn=5 precision=1.0000 recall=0.3750
- 开裂: gt=2 tp=2 fp=3 fn=0 precision=0.4000 recall=1.0000
- 油污: gt=18 tp=4 fp=11 fn=14 precision=0.2667 recall=0.2222
- 浅划伤: gt=13 tp=10 fp=7 fn=3 precision=0.5882 recall=0.7692
- 漏背锡: gt=46 tp=38 fp=57 fn=3 precision=0.4000 recall=0.8261
- 碰伤: gt=269 tp=190 fp=103 fn=71 precision=0.6485 recall=0.7063
- 脏污: gt=42 tp=26 fp=18 fn=15 precision=0.5909 recall=0.6190
- 轮廓划伤: gt=84 tp=54 fp=67 fn=24 precision=0.4463 recall=0.6429
- 锡丝残留: gt=12 tp=9 fp=9 fn=3 precision=0.5000 recall=0.7500
- 锡尖: gt=27 tp=20 fp=1 fn=7 precision=0.9524 recall=0.7407
- 锡膏: gt=46 tp=34 fp=18 fn=11 precision=0.6538 recall=0.7391
