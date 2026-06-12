# Validation Error Diagnosis

- Status: completed
- TP: 320
- FP: 78
- FN: 121
- Precision: 0.8040
- Recall: 0.7240

## Diagnosis Vector
- small_object_score: 0.3144
- low_contrast_score: 0.4744
- class_imbalance_score: 0.9603
- localization_score: 0.0023
- false_positive_score: 0.1960

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=38 fp=2 fn=0 precision=0.9500 recall=1.0000
- OK3: gt=102 tp=101 fp=22 fn=1 precision=0.8211 recall=0.9902
- 加强筋打伤: gt=6 tp=5 fp=0 fn=1 precision=1.0000 recall=0.8333
- 开裂: gt=3 tp=1 fp=0 fn=2 precision=1.0000 recall=0.3333
- 油污: gt=17 tp=2 fp=4 fn=15 precision=0.3333 recall=0.1176
- 浅划伤: gt=14 tp=2 fp=9 fn=12 precision=0.1818 recall=0.1429
- 漏背锡: gt=32 tp=20 fp=6 fn=11 precision=0.7692 recall=0.6250
- 碰伤: gt=133 tp=94 fp=25 fn=39 precision=0.7899 recall=0.7068
- 脏污: gt=28 tp=2 fp=1 fn=26 precision=0.6667 recall=0.0714
- 轮廓划伤: gt=30 tp=29 fp=5 fn=1 precision=0.8529 recall=0.9667
- 锡丝残留: gt=6 tp=3 fp=1 fn=3 precision=0.7500 recall=0.5000
- 锡尖: gt=12 tp=9 fp=2 fn=3 precision=0.8182 recall=0.7500
- 锡膏: gt=21 tp=14 fp=1 fn=7 precision=0.9333 recall=0.6667
