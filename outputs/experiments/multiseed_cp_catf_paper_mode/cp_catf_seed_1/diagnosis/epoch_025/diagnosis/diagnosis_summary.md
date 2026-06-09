# Validation Error Diagnosis

- Status: completed
- TP: 349
- FP: 160
- FN: 84
- Precision: 0.6857
- Recall: 0.7896

## Diagnosis Vector
- small_object_score: 0.2341
- low_contrast_score: 0.5274
- class_imbalance_score: 0.9853
- localization_score: 0.0204
- false_positive_score: 0.3143

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=38 fp=2 fn=0 precision=0.9500 recall=1.0000
- OK3: gt=102 tp=102 fp=24 fn=0 precision=0.8095 recall=1.0000
- 加强筋打伤: gt=6 tp=5 fp=0 fn=1 precision=1.0000 recall=0.8333
- 开裂: gt=3 tp=3 fp=7 fn=0 precision=0.3000 recall=1.0000
- 油污: gt=17 tp=2 fp=14 fn=15 precision=0.1250 recall=0.1176
- 浅划伤: gt=14 tp=0 fp=3 fn=14 precision=0.0000 recall=0.0000
- 漏背锡: gt=32 tp=21 fp=4 fn=10 precision=0.8400 recall=0.6562
- 碰伤: gt=133 tp=103 fp=34 fn=28 precision=0.7518 recall=0.7744
- 脏污: gt=28 tp=15 fp=26 fn=9 precision=0.3659 recall=0.5357
- 轮廓划伤: gt=30 tp=28 fp=27 fn=1 precision=0.5091 recall=0.9333
- 锡丝残留: gt=6 tp=5 fp=2 fn=1 precision=0.7143 recall=0.8333
- 锡尖: gt=12 tp=10 fp=6 fn=2 precision=0.6250 recall=0.8333
- 锡膏: gt=21 tp=17 fp=11 fn=3 precision=0.6071 recall=0.8095
