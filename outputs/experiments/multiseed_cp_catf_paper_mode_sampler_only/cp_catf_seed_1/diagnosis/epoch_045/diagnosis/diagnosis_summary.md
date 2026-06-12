# Validation Error Diagnosis

- Status: completed
- TP: 385
- FP: 111
- FN: 55
- Precision: 0.7762
- Recall: 0.8710

## Diagnosis Vector
- small_object_score: 0.1672
- low_contrast_score: 0.6409
- class_imbalance_score: 0.9030
- localization_score: 0.0045
- false_positive_score: 0.2238

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=38 fp=1 fn=0 precision=0.9744 recall=1.0000
- OK3: gt=102 tp=102 fp=21 fn=0 precision=0.8293 recall=1.0000
- 加强筋打伤: gt=6 tp=6 fp=1 fn=0 precision=0.8571 recall=1.0000
- 开裂: gt=3 tp=3 fp=2 fn=0 precision=0.6000 recall=1.0000
- 油污: gt=17 tp=4 fp=5 fn=12 precision=0.4444 recall=0.2353
- 浅划伤: gt=14 tp=5 fp=7 fn=8 precision=0.4167 recall=0.3571
- 漏背锡: gt=32 tp=30 fp=10 fn=2 precision=0.7500 recall=0.9375
- 碰伤: gt=133 tp=111 fp=24 fn=22 precision=0.8222 recall=0.8346
- 脏污: gt=28 tp=21 fp=26 fn=7 precision=0.4468 recall=0.7500
- 轮廓划伤: gt=30 tp=27 fp=6 fn=3 precision=0.8182 recall=0.9000
- 锡丝残留: gt=6 tp=6 fp=2 fn=0 precision=0.7500 recall=1.0000
- 锡尖: gt=12 tp=12 fp=3 fn=0 precision=0.8000 recall=1.0000
- 锡膏: gt=21 tp=20 fp=3 fn=1 precision=0.8696 recall=0.9524
