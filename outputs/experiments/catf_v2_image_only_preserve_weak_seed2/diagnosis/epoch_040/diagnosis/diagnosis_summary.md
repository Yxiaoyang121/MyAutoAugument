# Validation Error Diagnosis

- Status: completed
- TP: 721
- FP: 282
- FN: 167
- Precision: 0.7188
- Recall: 0.7967

## Diagnosis Vector
- small_object_score: 0.2479
- low_contrast_score: 0.5132
- class_imbalance_score: 0.9953
- localization_score: 0.0188
- false_positive_score: 0.2812

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=21 fn=0 precision=0.7529 recall=1.0000
- OK3: gt=274 tp=274 fp=69 fn=0 precision=0.7988 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=1 fn=1 precision=0.8750 recall=0.8750
- 开裂: gt=2 tp=0 fp=13 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=6 fp=14 fn=11 precision=0.3000 recall=0.3333
- 浅划伤: gt=13 tp=7 fp=4 fn=6 precision=0.6364 recall=0.5385
- 漏背锡: gt=46 tp=27 fp=15 fn=16 precision=0.6429 recall=0.5870
- 碰伤: gt=269 tp=196 fp=81 fn=69 precision=0.7076 recall=0.7286
- 脏污: gt=42 tp=27 fp=26 fn=10 precision=0.5094 recall=0.6429
- 轮廓划伤: gt=84 tp=46 fp=13 fn=34 precision=0.7797 recall=0.5476
- 锡丝残留: gt=12 tp=9 fp=2 fn=3 precision=0.8182 recall=0.7500
- 锡尖: gt=27 tp=23 fp=6 fn=4 precision=0.7931 recall=0.8519
- 锡膏: gt=46 tp=35 fp=17 fn=11 precision=0.6731 recall=0.7609
