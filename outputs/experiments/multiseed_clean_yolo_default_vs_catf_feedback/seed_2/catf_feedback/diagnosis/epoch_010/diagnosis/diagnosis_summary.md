# Validation Error Diagnosis

- Status: completed
- TP: 651
- FP: 324
- FN: 232
- Precision: 0.6677
- Recall: 0.7193

## Diagnosis Vector
- small_object_score: 0.3277
- low_contrast_score: 0.4679
- class_imbalance_score: 0.9953
- localization_score: 0.0243
- false_positive_score: 0.3323

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=21 fn=0 precision=0.7529 recall=1.0000
- OK3: gt=274 tp=266 fp=63 fn=8 precision=0.8085 recall=0.9708
- 加强筋打伤: gt=8 tp=0 fp=0 fn=8 precision=0.0000 recall=0.0000
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=4 fp=23 fn=13 precision=0.1481 recall=0.2222
- 浅划伤: gt=13 tp=4 fp=0 fn=9 precision=1.0000 recall=0.3077
- 漏背锡: gt=46 tp=17 fp=15 fn=23 precision=0.5312 recall=0.3696
- 碰伤: gt=269 tp=182 fp=110 fn=81 precision=0.6233 recall=0.6766
- 脏污: gt=42 tp=8 fp=4 fn=32 precision=0.6667 recall=0.1905
- 轮廓划伤: gt=84 tp=48 fp=63 fn=30 precision=0.4324 recall=0.5714
- 锡丝残留: gt=12 tp=0 fp=0 fn=12 precision=0.0000 recall=0.0000
- 锡尖: gt=27 tp=22 fp=0 fn=5 precision=1.0000 recall=0.8148
- 锡膏: gt=46 tp=36 fp=25 fn=9 precision=0.5902 recall=0.7826
