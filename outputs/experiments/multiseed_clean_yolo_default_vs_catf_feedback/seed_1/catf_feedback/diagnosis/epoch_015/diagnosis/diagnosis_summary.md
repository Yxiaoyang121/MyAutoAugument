# Validation Error Diagnosis

- Status: completed
- TP: 652
- FP: 289
- FN: 234
- Precision: 0.6929
- Recall: 0.7204

## Diagnosis Vector
- small_object_score: 0.3379
- low_contrast_score: 0.4624
- class_imbalance_score: 0.9953
- localization_score: 0.0210
- false_positive_score: 0.3071

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=20 fn=0 precision=0.7619 recall=1.0000
- OK3: gt=274 tp=274 fp=65 fn=0 precision=0.8083 recall=1.0000
- 加强筋打伤: gt=8 tp=6 fp=2 fn=1 precision=0.7500 recall=0.7500
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=5 fp=22 fn=13 precision=0.1852 recall=0.2778
- 浅划伤: gt=13 tp=5 fp=0 fn=8 precision=1.0000 recall=0.3846
- 漏背锡: gt=46 tp=17 fp=21 fn=23 precision=0.4474 recall=0.3696
- 碰伤: gt=269 tp=198 fp=119 fn=63 precision=0.6246 recall=0.7361
- 脏污: gt=42 tp=17 fp=26 fn=24 precision=0.3953 recall=0.4048
- 轮廓划伤: gt=84 tp=26 fp=10 fn=55 precision=0.7222 recall=0.3095
- 锡丝残留: gt=12 tp=1 fp=2 fn=11 precision=0.3333 recall=0.0833
- 锡尖: gt=27 tp=20 fp=2 fn=7 precision=0.9091 recall=0.7407
- 锡膏: gt=46 tp=19 fp=0 fn=27 precision=1.0000 recall=0.4130
