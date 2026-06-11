# Validation Error Diagnosis

- Status: completed
- TP: 375
- FP: 115
- FN: 62
- Precision: 0.7653
- Recall: 0.8484

## Diagnosis Vector
- small_object_score: 0.1739
- low_contrast_score: 0.6065
- class_imbalance_score: 0.9236
- localization_score: 0.0113
- false_positive_score: 0.2347

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=38 fp=2 fn=0 precision=0.9500 recall=1.0000
- OK3: gt=102 tp=102 fp=22 fn=0 precision=0.8226 recall=1.0000
- 加强筋打伤: gt=6 tp=6 fp=1 fn=0 precision=0.8571 recall=1.0000
- 开裂: gt=3 tp=3 fp=6 fn=0 precision=0.3333 recall=1.0000
- 油污: gt=17 tp=3 fp=3 fn=13 precision=0.5000 recall=0.1765
- 浅划伤: gt=14 tp=4 fp=12 fn=9 precision=0.2500 recall=0.2857
- 漏背锡: gt=32 tp=28 fp=10 fn=3 precision=0.7368 recall=0.8750
- 碰伤: gt=133 tp=110 fp=21 fn=23 precision=0.8397 recall=0.8271
- 脏污: gt=28 tp=17 fp=23 fn=9 precision=0.4250 recall=0.6071
- 轮廓划伤: gt=30 tp=29 fp=2 fn=1 precision=0.9355 recall=0.9667
- 锡丝残留: gt=6 tp=4 fp=2 fn=2 precision=0.6667 recall=0.6667
- 锡尖: gt=12 tp=11 fp=8 fn=1 precision=0.5789 recall=0.9167
- 锡膏: gt=21 tp=20 fp=3 fn=1 precision=0.8696 recall=0.9524
