# Validation Error Diagnosis

- Status: completed
- TP: 375
- FP: 102
- FN: 62
- Precision: 0.7862
- Recall: 0.8484

## Diagnosis Vector
- small_object_score: 0.1839
- low_contrast_score: 0.6153
- class_imbalance_score: 0.9603
- localization_score: 0.0113
- false_positive_score: 0.2138

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=38 fp=2 fn=0 precision=0.9500 recall=1.0000
- OK3: gt=102 tp=102 fp=22 fn=0 precision=0.8226 recall=1.0000
- 加强筋打伤: gt=6 tp=6 fp=0 fn=0 precision=1.0000 recall=1.0000
- 开裂: gt=3 tp=3 fp=1 fn=0 precision=0.7500 recall=1.0000
- 油污: gt=17 tp=5 fp=5 fn=12 precision=0.5000 recall=0.2941
- 浅划伤: gt=14 tp=1 fp=5 fn=13 precision=0.1667 recall=0.0714
- 漏背锡: gt=32 tp=29 fp=9 fn=3 precision=0.7632 recall=0.9062
- 碰伤: gt=133 tp=109 fp=22 fn=24 precision=0.8321 recall=0.8195
- 脏污: gt=28 tp=17 fp=16 fn=7 precision=0.5152 recall=0.6071
- 轮廓划伤: gt=30 tp=29 fp=4 fn=0 precision=0.8788 recall=0.9667
- 锡丝残留: gt=6 tp=4 fp=2 fn=2 precision=0.6667 recall=0.6667
- 锡尖: gt=12 tp=12 fp=8 fn=0 precision=0.6000 recall=1.0000
- 锡膏: gt=21 tp=20 fp=6 fn=1 precision=0.7692 recall=0.9524
