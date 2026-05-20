# Validation Error Diagnosis

- Status: completed
- TP: 665
- FP: 156
- FN: 228
- Precision: 0.8100
- Recall: 0.7348

## Diagnosis Vector
- small_object_score: 0.3413
- low_contrast_score: 0.5842
- class_imbalance_score: 0.9953
- localization_score: 0.0133
- false_positive_score: 0.1900

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=1 fn=0 precision=0.9846 recall=1.0000
- OK3: gt=274 tp=271 fp=10 fn=3 precision=0.9644 recall=0.9891
- 加强筋打伤: gt=8 tp=6 fp=1 fn=2 precision=0.8571 recall=0.7500
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=6 fp=11 fn=11 precision=0.3529 recall=0.3333
- 浅划伤: gt=13 tp=9 fp=12 fn=4 precision=0.4286 recall=0.6923
- 漏背锡: gt=46 tp=32 fp=19 fn=11 precision=0.6275 recall=0.6957
- 碰伤: gt=269 tp=138 fp=50 fn=126 precision=0.7340 recall=0.5130
- 脏污: gt=42 tp=22 fp=14 fn=20 precision=0.6111 recall=0.5238
- 轮廓划伤: gt=84 tp=51 fp=19 fn=30 precision=0.7286 recall=0.6071
- 锡丝残留: gt=12 tp=7 fp=0 fn=5 precision=1.0000 recall=0.5833
- 锡尖: gt=27 tp=27 fp=3 fn=0 precision=0.9000 recall=1.0000
- 锡膏: gt=46 tp=32 fp=16 fn=14 precision=0.6667 recall=0.6957
