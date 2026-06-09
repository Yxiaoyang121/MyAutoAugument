# Validation Error Diagnosis

- Status: completed
- TP: 366
- FP: 107
- FN: 65
- Precision: 0.7738
- Recall: 0.8281

## Diagnosis Vector
- small_object_score: 0.2007
- low_contrast_score: 0.5685
- class_imbalance_score: 0.9030
- localization_score: 0.0249
- false_positive_score: 0.2262

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=38 fp=3 fn=0 precision=0.9268 recall=1.0000
- OK3: gt=102 tp=102 fp=23 fn=0 precision=0.8160 recall=1.0000
- 加强筋打伤: gt=6 tp=6 fp=1 fn=0 precision=0.8571 recall=1.0000
- 开裂: gt=3 tp=3 fp=0 fn=0 precision=1.0000 recall=1.0000
- 油污: gt=17 tp=4 fp=3 fn=13 precision=0.5714 recall=0.2353
- 浅划伤: gt=14 tp=4 fp=8 fn=9 precision=0.3333 recall=0.2857
- 漏背锡: gt=32 tp=24 fp=13 fn=4 precision=0.6486 recall=0.7500
- 碰伤: gt=133 tp=109 fp=14 fn=22 precision=0.8862 recall=0.8195
- 脏污: gt=28 tp=16 fp=19 fn=8 precision=0.4571 recall=0.5714
- 轮廓划伤: gt=30 tp=28 fp=7 fn=2 precision=0.8000 recall=0.9333
- 锡丝残留: gt=6 tp=5 fp=4 fn=1 precision=0.5556 recall=0.8333
- 锡尖: gt=12 tp=9 fp=6 fn=3 precision=0.6000 recall=0.7500
- 锡膏: gt=21 tp=18 fp=6 fn=3 precision=0.7500 recall=0.8571
