# Validation Error Diagnosis

- Status: completed
- TP: 702
- FP: 324
- FN: 185
- Precision: 0.6842
- Recall: 0.7757

## Diagnosis Vector
- small_object_score: 0.2852
- low_contrast_score: 0.5319
- class_imbalance_score: 0.8980
- localization_score: 0.0199
- false_positive_score: 0.3158

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=24 fn=0 precision=0.7273 recall=1.0000
- OK3: gt=274 tp=273 fp=67 fn=1 precision=0.8029 recall=0.9964
- 加强筋打伤: gt=8 tp=7 fp=0 fn=1 precision=1.0000 recall=0.8750
- 开裂: gt=2 tp=2 fp=12 fn=0 precision=0.1429 recall=1.0000
- 油污: gt=18 tp=5 fp=24 fn=12 precision=0.1724 recall=0.2778
- 浅划伤: gt=13 tp=6 fp=2 fn=7 precision=0.7500 recall=0.4615
- 漏背锡: gt=46 tp=29 fp=21 fn=14 precision=0.5800 recall=0.6304
- 碰伤: gt=269 tp=173 fp=74 fn=91 precision=0.7004 recall=0.6431
- 脏污: gt=42 tp=27 fp=26 fn=12 precision=0.5094 recall=0.6429
- 轮廓划伤: gt=84 tp=52 fp=47 fn=28 precision=0.5253 recall=0.6190
- 锡丝残留: gt=12 tp=8 fp=3 fn=4 precision=0.7273 recall=0.6667
- 锡尖: gt=27 tp=23 fp=3 fn=4 precision=0.8846 recall=0.8519
- 锡膏: gt=46 tp=33 fp=21 fn=11 precision=0.6111 recall=0.7174
