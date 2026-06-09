# Validation Error Diagnosis

- Status: completed
- TP: 372
- FP: 124
- FN: 67
- Precision: 0.7500
- Recall: 0.8416

## Diagnosis Vector
- small_object_score: 0.1940
- low_contrast_score: 0.6112
- class_imbalance_score: 0.9030
- localization_score: 0.0068
- false_positive_score: 0.2500

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=38 fp=2 fn=0 precision=0.9500 recall=1.0000
- OK3: gt=102 tp=102 fp=24 fn=0 precision=0.8095 recall=1.0000
- 加强筋打伤: gt=6 tp=6 fp=1 fn=0 precision=0.8571 recall=1.0000
- 开裂: gt=3 tp=3 fp=8 fn=0 precision=0.2727 recall=1.0000
- 油污: gt=17 tp=4 fp=4 fn=13 precision=0.5000 recall=0.2353
- 浅划伤: gt=14 tp=4 fp=8 fn=9 precision=0.3333 recall=0.2857
- 漏背锡: gt=32 tp=27 fp=13 fn=4 precision=0.6750 recall=0.8438
- 碰伤: gt=133 tp=106 fp=27 fn=27 precision=0.7970 recall=0.7970
- 脏污: gt=28 tp=15 fp=11 fn=12 precision=0.5769 recall=0.5357
- 轮廓划伤: gt=30 tp=29 fp=16 fn=1 precision=0.6444 recall=0.9667
- 锡丝残留: gt=6 tp=6 fp=3 fn=0 precision=0.6667 recall=1.0000
- 锡尖: gt=12 tp=12 fp=2 fn=0 precision=0.8571 recall=1.0000
- 锡膏: gt=21 tp=20 fp=5 fn=1 precision=0.8000 recall=0.9524
