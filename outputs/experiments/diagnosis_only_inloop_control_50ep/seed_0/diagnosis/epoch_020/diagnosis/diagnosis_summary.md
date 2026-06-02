# Validation Error Diagnosis

- Status: completed
- TP: 689
- FP: 397
- FN: 205
- Precision: 0.6344
- Recall: 0.7613

## Diagnosis Vector
- small_object_score: 0.3090
- low_contrast_score: 0.5366
- class_imbalance_score: 0.9953
- localization_score: 0.0122
- false_positive_score: 0.3656

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=30 fn=0 precision=0.6809 recall=1.0000
- OK3: gt=274 tp=274 fp=70 fn=0 precision=0.7965 recall=1.0000
- 加强筋打伤: gt=8 tp=6 fp=0 fn=2 precision=1.0000 recall=0.7500
- 开裂: gt=2 tp=0 fp=1 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=5 fp=79 fn=12 precision=0.0595 recall=0.2778
- 浅划伤: gt=13 tp=4 fp=0 fn=9 precision=1.0000 recall=0.3077
- 漏背锡: gt=46 tp=36 fp=39 fn=9 precision=0.4800 recall=0.7826
- 碰伤: gt=269 tp=176 fp=79 fn=90 precision=0.6902 recall=0.6543
- 脏污: gt=42 tp=15 fp=22 fn=25 precision=0.4054 recall=0.3571
- 轮廓划伤: gt=84 tp=44 fp=35 fn=37 precision=0.5570 recall=0.5238
- 锡丝残留: gt=12 tp=7 fp=6 fn=5 precision=0.5385 recall=0.5833
- 锡尖: gt=27 tp=22 fp=6 fn=5 precision=0.7857 recall=0.8148
- 锡膏: gt=46 tp=36 fp=30 fn=9 precision=0.5455 recall=0.7826
