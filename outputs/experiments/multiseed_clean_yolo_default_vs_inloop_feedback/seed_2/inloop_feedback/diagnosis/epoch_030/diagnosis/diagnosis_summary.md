# Validation Error Diagnosis

- Status: completed
- TP: 673
- FP: 276
- FN: 213
- Precision: 0.7092
- Recall: 0.7436

## Diagnosis Vector
- small_object_score: 0.3379
- low_contrast_score: 0.5660
- class_imbalance_score: 0.9078
- localization_score: 0.0210
- false_positive_score: 0.2908

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=22 fn=0 precision=0.7442 recall=1.0000
- OK3: gt=274 tp=273 fp=69 fn=1 precision=0.7982 recall=0.9964
- 加强筋打伤: gt=8 tp=5 fp=6 fn=3 precision=0.4545 recall=0.6250
- 开裂: gt=2 tp=1 fp=3 fn=1 precision=0.2500 recall=0.5000
- 油污: gt=18 tp=6 fp=18 fn=12 precision=0.2500 recall=0.3333
- 浅划伤: gt=13 tp=8 fp=0 fn=5 precision=1.0000 recall=0.6154
- 漏背锡: gt=46 tp=28 fp=20 fn=12 precision=0.5833 recall=0.6087
- 碰伤: gt=269 tp=151 fp=51 fn=114 precision=0.7475 recall=0.5613
- 脏污: gt=42 tp=23 fp=18 fn=15 precision=0.5610 recall=0.5476
- 轮廓划伤: gt=84 tp=50 fp=42 fn=30 precision=0.5435 recall=0.5952
- 锡丝残留: gt=12 tp=3 fp=5 fn=9 precision=0.3750 recall=0.2500
- 锡尖: gt=27 tp=25 fp=5 fn=2 precision=0.8333 recall=0.9259
- 锡膏: gt=46 tp=36 fp=17 fn=9 precision=0.6792 recall=0.7826
