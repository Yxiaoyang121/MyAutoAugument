# Validation Error Diagnosis

- Status: completed
- TP: 354
- FP: 155
- FN: 81
- Precision: 0.6955
- Recall: 0.8009

## Diagnosis Vector
- small_object_score: 0.2341
- low_contrast_score: 0.5481
- class_imbalance_score: 0.9236
- localization_score: 0.0158
- false_positive_score: 0.3045

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=38 fp=2 fn=0 precision=0.9500 recall=1.0000
- OK3: gt=102 tp=100 fp=22 fn=2 precision=0.8197 recall=0.9804
- 加强筋打伤: gt=6 tp=6 fp=0 fn=0 precision=1.0000 recall=1.0000
- 开裂: gt=3 tp=2 fp=0 fn=1 precision=1.0000 recall=0.6667
- 油污: gt=17 tp=3 fp=7 fn=13 precision=0.3000 recall=0.1765
- 浅划伤: gt=14 tp=3 fp=0 fn=11 precision=1.0000 recall=0.2143
- 漏背锡: gt=32 tp=28 fp=32 fn=4 precision=0.4667 recall=0.8750
- 碰伤: gt=133 tp=103 fp=32 fn=30 precision=0.7630 recall=0.7744
- 脏污: gt=28 tp=13 fp=16 fn=12 precision=0.4483 recall=0.4643
- 轮廓划伤: gt=30 tp=29 fp=19 fn=0 precision=0.6042 recall=0.9667
- 锡丝残留: gt=6 tp=3 fp=8 fn=2 precision=0.2727 recall=0.5000
- 锡尖: gt=12 tp=11 fp=9 fn=1 precision=0.5500 recall=0.9167
- 锡膏: gt=21 tp=15 fp=8 fn=5 precision=0.6522 recall=0.7143
