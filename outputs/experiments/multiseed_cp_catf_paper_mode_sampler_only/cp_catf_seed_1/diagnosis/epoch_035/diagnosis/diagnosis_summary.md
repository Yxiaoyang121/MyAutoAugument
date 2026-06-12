# Validation Error Diagnosis

- Status: completed
- TP: 367
- FP: 109
- FN: 71
- Precision: 0.7710
- Recall: 0.8303

## Diagnosis Vector
- small_object_score: 0.2107
- low_contrast_score: 0.5690
- class_imbalance_score: 0.9103
- localization_score: 0.0090
- false_positive_score: 0.2290

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=38 fp=2 fn=0 precision=0.9500 recall=1.0000
- OK3: gt=102 tp=101 fp=23 fn=1 precision=0.8145 recall=0.9902
- 加强筋打伤: gt=6 tp=6 fp=0 fn=0 precision=1.0000 recall=1.0000
- 开裂: gt=3 tp=3 fp=4 fn=0 precision=0.4286 recall=1.0000
- 油污: gt=17 tp=4 fp=4 fn=13 precision=0.5000 recall=0.2353
- 浅划伤: gt=14 tp=3 fp=2 fn=11 precision=0.6000 recall=0.2143
- 漏背锡: gt=32 tp=24 fp=12 fn=6 precision=0.6667 recall=0.7500
- 碰伤: gt=133 tp=107 fp=17 fn=25 precision=0.8629 recall=0.8045
- 脏污: gt=28 tp=15 fp=21 fn=12 precision=0.4167 recall=0.5357
- 轮廓划伤: gt=30 tp=29 fp=13 fn=1 precision=0.6905 recall=0.9667
- 锡丝残留: gt=6 tp=6 fp=3 fn=0 precision=0.6667 recall=1.0000
- 锡尖: gt=12 tp=12 fp=0 fn=0 precision=1.0000 recall=1.0000
- 锡膏: gt=21 tp=19 fp=8 fn=2 precision=0.7037 recall=0.9048
