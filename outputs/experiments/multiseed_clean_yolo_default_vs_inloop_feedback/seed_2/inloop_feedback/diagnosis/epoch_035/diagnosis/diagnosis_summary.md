# Validation Error Diagnosis

- Status: completed
- TP: 718
- FP: 290
- FN: 168
- Precision: 0.7123
- Recall: 0.7934

## Diagnosis Vector
- small_object_score: 0.2513
- low_contrast_score: 0.5051
- class_imbalance_score: 0.9758
- localization_score: 0.0210
- false_positive_score: 0.2877

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=20 fn=0 precision=0.7619 recall=1.0000
- OK3: gt=274 tp=272 fp=69 fn=2 precision=0.7977 recall=0.9927
- 加强筋打伤: gt=8 tp=6 fp=1 fn=2 precision=0.8571 recall=0.7500
- 开裂: gt=2 tp=1 fp=5 fn=1 precision=0.1667 recall=0.5000
- 油污: gt=18 tp=1 fp=9 fn=17 precision=0.1000 recall=0.0556
- 浅划伤: gt=13 tp=7 fp=3 fn=6 precision=0.7000 recall=0.5385
- 漏背锡: gt=46 tp=35 fp=21 fn=11 precision=0.6250 recall=0.7609
- 碰伤: gt=269 tp=200 fp=81 fn=63 precision=0.7117 recall=0.7435
- 脏污: gt=42 tp=19 fp=14 fn=18 precision=0.5758 recall=0.4524
- 轮廓划伤: gt=84 tp=48 fp=40 fn=29 precision=0.5455 recall=0.5714
- 锡丝残留: gt=12 tp=9 fp=9 fn=3 precision=0.5000 recall=0.7500
- 锡尖: gt=27 tp=24 fp=8 fn=3 precision=0.7500 recall=0.8889
- 锡膏: gt=46 tp=32 fp=10 fn=13 precision=0.7619 recall=0.6957
