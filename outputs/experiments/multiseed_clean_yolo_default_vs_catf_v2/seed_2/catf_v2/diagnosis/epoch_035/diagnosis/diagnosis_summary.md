# Validation Error Diagnosis

- Status: completed
- TP: 706
- FP: 291
- FN: 175
- Precision: 0.7081
- Recall: 0.7801

## Diagnosis Vector
- small_object_score: 0.2649
- low_contrast_score: 0.5117
- class_imbalance_score: 0.9564
- localization_score: 0.0265
- false_positive_score: 0.2919

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=20 fn=0 precision=0.7619 recall=1.0000
- OK3: gt=274 tp=274 fp=70 fn=0 precision=0.7965 recall=1.0000
- 加强筋打伤: gt=8 tp=6 fp=1 fn=2 precision=0.8571 recall=0.7500
- 开裂: gt=2 tp=2 fp=2 fn=0 precision=0.5000 recall=1.0000
- 油污: gt=18 tp=2 fp=13 fn=15 precision=0.1333 recall=0.1111
- 浅划伤: gt=13 tp=4 fp=1 fn=9 precision=0.8000 recall=0.3077
- 漏背锡: gt=46 tp=29 fp=20 fn=6 precision=0.5918 recall=0.6304
- 碰伤: gt=269 tp=192 fp=89 fn=72 precision=0.6833 recall=0.7138
- 脏污: gt=42 tp=19 fp=18 fn=19 precision=0.5135 recall=0.4524
- 轮廓划伤: gt=84 tp=50 fp=17 fn=31 precision=0.7463 recall=0.5952
- 锡丝残留: gt=12 tp=9 fp=10 fn=3 precision=0.4737 recall=0.7500
- 锡尖: gt=27 tp=23 fp=7 fn=4 precision=0.7667 recall=0.8519
- 锡膏: gt=46 tp=32 fp=23 fn=14 precision=0.5818 recall=0.6957
