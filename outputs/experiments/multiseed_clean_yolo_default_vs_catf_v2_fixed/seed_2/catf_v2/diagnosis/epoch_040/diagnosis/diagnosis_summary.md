# Validation Error Diagnosis

- Status: completed
- TP: 726
- FP: 310
- FN: 163
- Precision: 0.7008
- Recall: 0.8022

## Diagnosis Vector
- small_object_score: 0.2445
- low_contrast_score: 0.5482
- class_imbalance_score: 0.8536
- localization_score: 0.0177
- false_positive_score: 0.2992

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=24 fn=0 precision=0.7273 recall=1.0000
- OK3: gt=274 tp=274 fp=70 fn=0 precision=0.7965 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=0 fn=1 precision=1.0000 recall=0.8750
- 开裂: gt=2 tp=2 fp=7 fn=0 precision=0.2222 recall=1.0000
- 油污: gt=18 tp=8 fp=29 fn=7 precision=0.2162 recall=0.4444
- 浅划伤: gt=13 tp=8 fp=1 fn=5 precision=0.8889 recall=0.6154
- 漏背锡: gt=46 tp=30 fp=30 fn=10 precision=0.5000 recall=0.6522
- 碰伤: gt=269 tp=204 fp=93 fn=62 precision=0.6869 recall=0.7584
- 脏污: gt=42 tp=27 fp=18 fn=15 precision=0.6000 recall=0.6429
- 轮廓划伤: gt=84 tp=34 fp=15 fn=46 precision=0.6939 recall=0.4048
- 锡丝残留: gt=12 tp=9 fp=3 fn=3 precision=0.7500 recall=0.7500
- 锡尖: gt=27 tp=27 fp=8 fn=0 precision=0.7714 recall=1.0000
- 锡膏: gt=46 tp=32 fp=12 fn=14 precision=0.7273 recall=0.6957
