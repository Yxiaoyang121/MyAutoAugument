# Validation Error Diagnosis

- Status: completed
- TP: 742
- FP: 300
- FN: 148
- Precision: 0.7121
- Recall: 0.8199

## Diagnosis Vector
- small_object_score: 0.2377
- low_contrast_score: 0.5321
- class_imbalance_score: 0.8591
- localization_score: 0.0166
- false_positive_score: 0.2879

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=20 fn=0 precision=0.7619 recall=1.0000
- OK3: gt=274 tp=274 fp=71 fn=0 precision=0.7942 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=1 fn=1 precision=0.8750 recall=0.8750
- 开裂: gt=2 tp=2 fp=12 fn=0 precision=0.1429 recall=1.0000
- 油污: gt=18 tp=7 fp=5 fn=10 precision=0.5833 recall=0.3889
- 浅划伤: gt=13 tp=10 fp=15 fn=3 precision=0.4000 recall=0.7692
- 漏背锡: gt=46 tp=36 fp=27 fn=8 precision=0.5714 recall=0.7826
- 碰伤: gt=269 tp=199 fp=72 fn=62 precision=0.7343 recall=0.7398
- 脏污: gt=42 tp=24 fp=26 fn=17 precision=0.4800 recall=0.5714
- 轮廓划伤: gt=84 tp=52 fp=27 fn=31 precision=0.6582 recall=0.6190
- 锡丝残留: gt=12 tp=10 fp=8 fn=2 precision=0.5556 recall=0.8333
- 锡尖: gt=27 tp=27 fp=4 fn=0 precision=0.8710 recall=1.0000
- 锡膏: gt=46 tp=30 fp=12 fn=14 precision=0.7143 recall=0.6522
