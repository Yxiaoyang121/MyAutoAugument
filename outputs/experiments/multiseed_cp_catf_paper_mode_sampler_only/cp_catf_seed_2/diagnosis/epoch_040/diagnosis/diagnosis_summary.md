# Validation Error Diagnosis

- Status: completed
- TP: 377
- FP: 132
- FN: 65
- Precision: 0.7407
- Recall: 0.8529

## Diagnosis Vector
- small_object_score: 0.1873
- low_contrast_score: 0.5900
- class_imbalance_score: 0.8353
- localization_score: 0.0000
- false_positive_score: 0.2593

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting

## Per-Class Summary
- OK2: gt=38 tp=38 fp=2 fn=0 precision=0.9500 recall=1.0000
- OK3: gt=102 tp=102 fp=24 fn=0 precision=0.8095 recall=1.0000
- 加强筋打伤: gt=6 tp=6 fp=0 fn=0 precision=1.0000 recall=1.0000
- 开裂: gt=3 tp=3 fp=0 fn=0 precision=1.0000 recall=1.0000
- 油污: gt=17 tp=8 fp=14 fn=9 precision=0.3636 recall=0.4706
- 浅划伤: gt=14 tp=6 fp=4 fn=8 precision=0.6000 recall=0.4286
- 漏背锡: gt=32 tp=26 fp=11 fn=6 precision=0.7027 recall=0.8125
- 碰伤: gt=133 tp=105 fp=20 fn=28 precision=0.8400 recall=0.7895
- 脏污: gt=28 tp=18 fp=25 fn=10 precision=0.4186 recall=0.6429
- 轮廓划伤: gt=30 tp=29 fp=6 fn=1 precision=0.8286 recall=0.9667
- 锡丝残留: gt=6 tp=6 fp=3 fn=0 precision=0.6667 recall=1.0000
- 锡尖: gt=12 tp=12 fp=11 fn=0 precision=0.5217 recall=1.0000
- 锡膏: gt=21 tp=18 fp=12 fn=3 precision=0.6000 recall=0.8571
