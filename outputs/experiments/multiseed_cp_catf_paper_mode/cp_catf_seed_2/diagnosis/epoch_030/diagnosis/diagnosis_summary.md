# Validation Error Diagnosis

- Status: completed
- TP: 345
- FP: 88
- FN: 95
- Precision: 0.7968
- Recall: 0.7805

## Diagnosis Vector
- small_object_score: 0.2475
- low_contrast_score: 0.5068
- class_imbalance_score: 0.9853
- localization_score: 0.0045
- false_positive_score: 0.2032

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=38 fp=2 fn=0 precision=0.9500 recall=1.0000
- OK3: gt=102 tp=101 fp=22 fn=1 precision=0.8211 recall=0.9902
- 加强筋打伤: gt=6 tp=6 fp=1 fn=0 precision=0.8571 recall=1.0000
- 开裂: gt=3 tp=3 fp=3 fn=0 precision=0.5000 recall=1.0000
- 油污: gt=17 tp=3 fp=11 fn=14 precision=0.2143 recall=0.1765
- 浅划伤: gt=14 tp=0 fp=1 fn=14 precision=0.0000 recall=0.0000
- 漏背锡: gt=32 tp=22 fp=4 fn=10 precision=0.8462 recall=0.6875
- 碰伤: gt=133 tp=99 fp=16 fn=33 precision=0.8609 recall=0.7444
- 脏污: gt=28 tp=15 fp=10 fn=12 precision=0.6000 recall=0.5357
- 轮廓划伤: gt=30 tp=29 fp=7 fn=1 precision=0.8056 recall=0.9667
- 锡丝残留: gt=6 tp=3 fp=2 fn=3 precision=0.6000 recall=0.5000
- 锡尖: gt=12 tp=10 fp=5 fn=2 precision=0.6667 recall=0.8333
- 锡膏: gt=21 tp=16 fp=4 fn=5 precision=0.8000 recall=0.7619
