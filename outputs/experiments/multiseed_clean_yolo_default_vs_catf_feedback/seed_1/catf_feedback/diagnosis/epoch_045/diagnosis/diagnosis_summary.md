# Validation Error Diagnosis

- Status: completed
- TP: 746
- FP: 303
- FN: 140
- Precision: 0.7112
- Recall: 0.8243

## Diagnosis Vector
- small_object_score: 0.2326
- low_contrast_score: 0.5546
- class_imbalance_score: 0.8591
- localization_score: 0.0210
- false_positive_score: 0.2888

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=20 fn=0 precision=0.7619 recall=1.0000
- OK3: gt=274 tp=274 fp=63 fn=0 precision=0.8131 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=1 fn=1 precision=0.8750 recall=0.8750
- 开裂: gt=2 tp=2 fp=1 fn=0 precision=0.6667 recall=1.0000
- 油污: gt=18 tp=7 fp=26 fn=10 precision=0.2121 recall=0.3889
- 浅划伤: gt=13 tp=9 fp=14 fn=4 precision=0.3913 recall=0.6923
- 漏背锡: gt=46 tp=36 fp=23 fn=6 precision=0.6102 recall=0.7826
- 碰伤: gt=269 tp=194 fp=76 fn=69 precision=0.7185 recall=0.7212
- 脏污: gt=42 tp=27 fp=15 fn=11 precision=0.6429 recall=0.6429
- 轮廓划伤: gt=84 tp=57 fp=38 fn=23 precision=0.6000 recall=0.6786
- 锡丝残留: gt=12 tp=12 fp=7 fn=0 precision=0.6316 recall=1.0000
- 锡尖: gt=27 tp=25 fp=5 fn=2 precision=0.8333 recall=0.9259
- 锡膏: gt=46 tp=32 fp=14 fn=14 precision=0.6957 recall=0.6957
