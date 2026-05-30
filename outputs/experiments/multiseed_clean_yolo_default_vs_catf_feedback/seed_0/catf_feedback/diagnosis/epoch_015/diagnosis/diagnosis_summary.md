# Validation Error Diagnosis

- Status: completed
- TP: 665
- FP: 301
- FN: 220
- Precision: 0.6884
- Recall: 0.7348

## Diagnosis Vector
- small_object_score: 0.3362
- low_contrast_score: 0.5191
- class_imbalance_score: 0.9953
- localization_score: 0.0221
- false_positive_score: 0.3116

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=21 fn=0 precision=0.7529 recall=1.0000
- OK3: gt=274 tp=274 fp=65 fn=0 precision=0.8083 recall=1.0000
- 加强筋打伤: gt=8 tp=6 fp=1 fn=2 precision=0.8571 recall=0.7500
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=5 fp=27 fn=12 precision=0.1562 recall=0.2778
- 浅划伤: gt=13 tp=6 fp=0 fn=7 precision=1.0000 recall=0.4615
- 漏背锡: gt=46 tp=27 fp=33 fn=11 precision=0.4500 recall=0.5870
- 碰伤: gt=269 tp=172 fp=81 fn=93 precision=0.6798 recall=0.6394
- 脏污: gt=42 tp=18 fp=34 fn=20 precision=0.3462 recall=0.4286
- 轮廓划伤: gt=84 tp=43 fp=13 fn=38 precision=0.7679 recall=0.5119
- 锡丝残留: gt=12 tp=3 fp=5 fn=9 precision=0.3750 recall=0.2500
- 锡尖: gt=27 tp=19 fp=4 fn=8 precision=0.8261 recall=0.7037
- 锡膏: gt=46 tp=28 fp=17 fn=18 precision=0.6222 recall=0.6087
