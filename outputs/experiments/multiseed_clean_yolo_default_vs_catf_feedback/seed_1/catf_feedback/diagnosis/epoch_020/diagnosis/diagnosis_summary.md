# Validation Error Diagnosis

- Status: completed
- TP: 729
- FP: 471
- FN: 152
- Precision: 0.6075
- Recall: 0.8055

## Diagnosis Vector
- small_object_score: 0.2411
- low_contrast_score: 0.5477
- class_imbalance_score: 0.9953
- localization_score: 0.0265
- false_positive_score: 0.3925

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=32 fn=0 precision=0.6667 recall=1.0000
- OK3: gt=274 tp=271 fp=67 fn=3 precision=0.8018 recall=0.9891
- 加强筋打伤: gt=8 tp=6 fp=1 fn=2 precision=0.8571 recall=0.7500
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=3 fp=27 fn=14 precision=0.1000 recall=0.1667
- 浅划伤: gt=13 tp=8 fp=34 fn=5 precision=0.1905 recall=0.6154
- 漏背锡: gt=46 tp=31 fp=24 fn=12 precision=0.5636 recall=0.6739
- 碰伤: gt=269 tp=199 fp=96 fn=61 precision=0.6746 recall=0.7398
- 脏污: gt=42 tp=32 fp=85 fn=7 precision=0.2735 recall=0.7619
- 轮廓划伤: gt=84 tp=49 fp=47 fn=29 precision=0.5104 recall=0.5833
- 锡丝残留: gt=12 tp=3 fp=3 fn=9 precision=0.5000 recall=0.2500
- 锡尖: gt=27 tp=26 fp=5 fn=1 precision=0.8387 recall=0.9630
- 锡膏: gt=46 tp=37 fp=50 fn=7 precision=0.4253 recall=0.8043
