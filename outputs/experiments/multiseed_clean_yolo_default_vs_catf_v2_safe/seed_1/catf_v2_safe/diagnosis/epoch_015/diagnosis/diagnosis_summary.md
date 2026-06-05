# Validation Error Diagnosis

- Status: completed
- TP: 697
- FP: 371
- FN: 189
- Precision: 0.6526
- Recall: 0.7702

## Diagnosis Vector
- small_object_score: 0.2750
- low_contrast_score: 0.4479
- class_imbalance_score: 0.9953
- localization_score: 0.0210
- false_positive_score: 0.3474

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=23 fn=0 precision=0.7356 recall=1.0000
- OK3: gt=274 tp=274 fp=69 fn=0 precision=0.7988 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=2 fn=1 precision=0.7778 recall=0.8750
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=4 fp=38 fn=14 precision=0.0952 recall=0.2222
- 浅划伤: gt=13 tp=3 fp=0 fn=10 precision=1.0000 recall=0.2308
- 漏背锡: gt=46 tp=20 fp=14 fn=23 precision=0.5882 recall=0.4348
- 碰伤: gt=269 tp=200 fp=155 fn=61 precision=0.5634 recall=0.7435
- 脏污: gt=42 tp=15 fp=15 fn=25 precision=0.5000 recall=0.3571
- 轮廓划伤: gt=84 tp=52 fp=43 fn=26 precision=0.5474 recall=0.6190
- 锡丝残留: gt=12 tp=3 fp=1 fn=9 precision=0.7500 recall=0.2500
- 锡尖: gt=27 tp=23 fp=3 fn=4 precision=0.8846 recall=0.8519
- 锡膏: gt=46 tp=32 fp=8 fn=14 precision=0.8000 recall=0.6957
