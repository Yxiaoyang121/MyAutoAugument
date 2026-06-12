# Validation Error Diagnosis

- Status: completed
- TP: 359
- FP: 136
- FN: 78
- Precision: 0.7253
- Recall: 0.8122

## Diagnosis Vector
- small_object_score: 0.2274
- low_contrast_score: 0.5609
- class_imbalance_score: 0.9103
- localization_score: 0.0113
- false_positive_score: 0.2747

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
- 开裂: gt=3 tp=3 fp=2 fn=0 precision=0.6000 recall=1.0000
- 油污: gt=17 tp=4 fp=14 fn=13 precision=0.2222 recall=0.2353
- 浅划伤: gt=14 tp=3 fp=5 fn=11 precision=0.3750 recall=0.2143
- 漏背锡: gt=32 tp=26 fp=6 fn=4 precision=0.8125 recall=0.8125
- 碰伤: gt=133 tp=105 fp=30 fn=27 precision=0.7778 recall=0.7895
- 脏污: gt=28 tp=12 fp=20 fn=15 precision=0.3750 recall=0.4286
- 轮廓划伤: gt=30 tp=29 fp=17 fn=0 precision=0.6304 recall=0.9667
- 锡丝残留: gt=6 tp=4 fp=3 fn=2 precision=0.5714 recall=0.6667
- 锡尖: gt=12 tp=11 fp=7 fn=1 precision=0.6111 recall=0.9167
- 锡膏: gt=21 tp=18 fp=8 fn=3 precision=0.6923 recall=0.8571
