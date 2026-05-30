# Validation Error Diagnosis

- Status: completed
- TP: 708
- FP: 316
- FN: 179
- Precision: 0.6914
- Recall: 0.7823

## Diagnosis Vector
- small_object_score: 0.2869
- low_contrast_score: 0.5182
- class_imbalance_score: 0.8980
- localization_score: 0.0199
- false_positive_score: 0.3086

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=20 fn=0 precision=0.7619 recall=1.0000
- OK3: gt=274 tp=272 fp=72 fn=2 precision=0.7907 recall=0.9927
- 加强筋打伤: gt=8 tp=5 fp=1 fn=3 precision=0.8333 recall=0.6250
- 开裂: gt=2 tp=2 fp=1 fn=0 precision=0.6667 recall=1.0000
- 油污: gt=18 tp=5 fp=18 fn=13 precision=0.2174 recall=0.2778
- 浅划伤: gt=13 tp=7 fp=25 fn=5 precision=0.2188 recall=0.5385
- 漏背锡: gt=46 tp=37 fp=37 fn=6 precision=0.5000 recall=0.8043
- 碰伤: gt=269 tp=175 fp=76 fn=89 precision=0.6972 recall=0.6506
- 脏污: gt=42 tp=25 fp=17 fn=13 precision=0.5952 recall=0.5952
- 轮廓划伤: gt=84 tp=61 fp=30 fn=18 precision=0.6703 recall=0.7262
- 锡丝残留: gt=12 tp=10 fp=8 fn=2 precision=0.5556 recall=0.8333
- 锡尖: gt=27 tp=17 fp=3 fn=10 precision=0.8500 recall=0.6296
- 锡膏: gt=46 tp=28 fp=8 fn=18 precision=0.7778 recall=0.6087
