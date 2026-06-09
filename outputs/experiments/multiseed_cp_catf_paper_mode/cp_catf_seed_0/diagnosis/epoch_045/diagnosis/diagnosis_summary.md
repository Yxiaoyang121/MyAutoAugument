# Validation Error Diagnosis

- Status: completed
- TP: 369
- FP: 102
- FN: 67
- Precision: 0.7834
- Recall: 0.8348

## Diagnosis Vector
- small_object_score: 0.1973
- low_contrast_score: 0.6075
- class_imbalance_score: 0.9353
- localization_score: 0.0136
- false_positive_score: 0.2166

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=38 fp=1 fn=0 precision=0.9744 recall=1.0000
- OK3: gt=102 tp=102 fp=21 fn=0 precision=0.8293 recall=1.0000
- 加强筋打伤: gt=6 tp=6 fp=0 fn=0 precision=1.0000 recall=1.0000
- 开裂: gt=3 tp=3 fp=6 fn=0 precision=0.3333 recall=1.0000
- 油污: gt=17 tp=4 fp=4 fn=12 precision=0.5000 recall=0.2353
- 浅划伤: gt=14 tp=2 fp=8 fn=11 precision=0.2000 recall=0.1429
- 漏背锡: gt=32 tp=25 fp=10 fn=6 precision=0.7143 recall=0.7812
- 碰伤: gt=133 tp=104 fp=24 fn=29 precision=0.8125 recall=0.7820
- 脏污: gt=28 tp=20 fp=15 fn=6 precision=0.5714 recall=0.7143
- 轮廓划伤: gt=30 tp=29 fp=2 fn=0 precision=0.9355 recall=0.9667
- 锡丝残留: gt=6 tp=4 fp=2 fn=2 precision=0.6667 recall=0.6667
- 锡尖: gt=12 tp=12 fp=8 fn=0 precision=0.6000 recall=1.0000
- 锡膏: gt=21 tp=20 fp=1 fn=1 precision=0.9524 recall=0.9524
