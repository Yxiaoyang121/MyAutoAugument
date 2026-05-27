# Validation Error Diagnosis

- Status: completed
- TP: 725
- FP: 314
- FN: 152
- Precision: 0.6978
- Recall: 0.8011

## Diagnosis Vector
- small_object_score: 0.2445
- low_contrast_score: 0.5424
- class_imbalance_score: 0.8068
- localization_score: 0.0309
- false_positive_score: 0.3022

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=21 fn=0 precision=0.7529 recall=1.0000
- OK3: gt=274 tp=274 fp=67 fn=0 precision=0.8035 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=1 fn=1 precision=0.8750 recall=0.8750
- 开裂: gt=2 tp=2 fp=10 fn=0 precision=0.1667 recall=1.0000
- 油污: gt=18 tp=10 fp=23 fn=6 precision=0.3030 recall=0.5556
- 浅划伤: gt=13 tp=7 fp=3 fn=6 precision=0.7000 recall=0.5385
- 漏背锡: gt=46 tp=30 fp=21 fn=10 precision=0.5882 recall=0.6522
- 碰伤: gt=269 tp=190 fp=82 fn=69 precision=0.6985 recall=0.7063
- 脏污: gt=42 tp=23 fp=22 fn=15 precision=0.5111 recall=0.5476
- 轮廓划伤: gt=84 tp=50 fp=29 fn=30 precision=0.6329 recall=0.5952
- 锡丝残留: gt=12 tp=9 fp=6 fn=3 precision=0.6000 recall=0.7500
- 锡尖: gt=27 tp=24 fp=5 fn=3 precision=0.8276 recall=0.8889
- 锡膏: gt=46 tp=35 fp=24 fn=9 precision=0.5932 recall=0.7609
