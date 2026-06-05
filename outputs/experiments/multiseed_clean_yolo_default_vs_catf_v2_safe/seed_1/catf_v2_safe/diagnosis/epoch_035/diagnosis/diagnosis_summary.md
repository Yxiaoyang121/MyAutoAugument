# Validation Error Diagnosis

- Status: completed
- TP: 702
- FP: 280
- FN: 184
- Precision: 0.7149
- Recall: 0.7757

## Diagnosis Vector
- small_object_score: 0.2801
- low_contrast_score: 0.5372
- class_imbalance_score: 0.8786
- localization_score: 0.0210
- false_positive_score: 0.2851

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=20 fn=0 precision=0.7619 recall=1.0000
- OK3: gt=274 tp=274 fp=70 fn=0 precision=0.7965 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=3 fn=1 precision=0.7000 recall=0.8750
- 开裂: gt=2 tp=2 fp=1 fn=0 precision=0.6667 recall=1.0000
- 油污: gt=18 tp=6 fp=17 fn=12 precision=0.2609 recall=0.3333
- 浅划伤: gt=13 tp=8 fp=6 fn=5 precision=0.5714 recall=0.6154
- 漏背锡: gt=46 tp=29 fp=30 fn=7 precision=0.4915 recall=0.6304
- 碰伤: gt=269 tp=179 fp=74 fn=85 precision=0.7075 recall=0.6654
- 脏污: gt=42 tp=19 fp=9 fn=23 precision=0.6786 recall=0.4524
- 轮廓划伤: gt=84 tp=49 fp=33 fn=31 precision=0.5976 recall=0.5833
- 锡丝残留: gt=12 tp=11 fp=7 fn=1 precision=0.6111 recall=0.9167
- 锡尖: gt=27 tp=22 fp=3 fn=5 precision=0.8800 recall=0.8148
- 锡膏: gt=46 tp=32 fp=7 fn=14 precision=0.8205 recall=0.6957
