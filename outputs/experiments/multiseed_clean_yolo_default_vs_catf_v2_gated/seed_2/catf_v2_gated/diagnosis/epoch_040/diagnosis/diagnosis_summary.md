# Validation Error Diagnosis

- Status: completed
- TP: 721
- FP: 303
- FN: 166
- Precision: 0.7041
- Recall: 0.7967

## Diagnosis Vector
- small_object_score: 0.2428
- low_contrast_score: 0.5358
- class_imbalance_score: 0.9175
- localization_score: 0.0199
- false_positive_score: 0.2959

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=22 fn=0 precision=0.7442 recall=1.0000
- OK3: gt=274 tp=274 fp=70 fn=0 precision=0.7965 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=0 fn=1 precision=1.0000 recall=0.8750
- 开裂: gt=2 tp=2 fp=14 fn=0 precision=0.1250 recall=1.0000
- 油污: gt=18 tp=4 fp=17 fn=12 precision=0.1905 recall=0.2222
- 浅划伤: gt=13 tp=8 fp=4 fn=5 precision=0.6667 recall=0.6154
- 漏背锡: gt=46 tp=26 fp=19 fn=13 precision=0.5778 recall=0.5652
- 碰伤: gt=269 tp=199 fp=99 fn=66 precision=0.6678 recall=0.7398
- 脏污: gt=42 tp=30 fp=19 fn=11 precision=0.6122 recall=0.7143
- 轮廓划伤: gt=84 tp=37 fp=19 fn=43 precision=0.6607 recall=0.4405
- 锡丝残留: gt=12 tp=9 fp=5 fn=3 precision=0.6429 recall=0.7500
- 锡尖: gt=27 tp=27 fp=4 fn=0 precision=0.8710 recall=1.0000
- 锡膏: gt=46 tp=34 fp=11 fn=12 precision=0.7556 recall=0.7391
