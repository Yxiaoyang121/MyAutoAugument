# Validation Error Diagnosis

- Status: completed
- TP: 371
- FP: 102
- FN: 64
- Precision: 0.7844
- Recall: 0.8394

## Diagnosis Vector
- small_object_score: 0.1839
- low_contrast_score: 0.5406
- class_imbalance_score: 0.8824
- localization_score: 0.0158
- false_positive_score: 0.2156

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=38 fp=3 fn=0 precision=0.9268 recall=1.0000
- OK3: gt=102 tp=102 fp=23 fn=0 precision=0.8160 recall=1.0000
- 加强筋打伤: gt=6 tp=6 fp=1 fn=0 precision=0.8571 recall=1.0000
- 开裂: gt=3 tp=2 fp=1 fn=1 precision=0.6667 recall=0.6667
- 油污: gt=17 tp=5 fp=3 fn=12 precision=0.6250 recall=0.2941
- 浅划伤: gt=14 tp=10 fp=11 fn=2 precision=0.4762 recall=0.7143
- 漏背锡: gt=32 tp=28 fp=10 fn=4 precision=0.7368 recall=0.8750
- 碰伤: gt=133 tp=107 fp=24 fn=25 precision=0.8168 recall=0.8045
- 脏污: gt=28 tp=14 fp=12 fn=11 precision=0.5385 recall=0.5000
- 轮廓划伤: gt=30 tp=28 fp=4 fn=1 precision=0.8750 recall=0.9333
- 锡丝残留: gt=6 tp=3 fp=2 fn=3 precision=0.6000 recall=0.5000
- 锡尖: gt=12 tp=12 fp=8 fn=0 precision=0.6000 recall=1.0000
- 锡膏: gt=21 tp=16 fp=0 fn=5 precision=1.0000 recall=0.7619
