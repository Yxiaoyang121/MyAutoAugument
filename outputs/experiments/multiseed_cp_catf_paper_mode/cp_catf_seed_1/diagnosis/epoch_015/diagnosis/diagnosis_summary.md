# Validation Error Diagnosis

- Status: completed
- TP: 330
- FP: 147
- FN: 108
- Precision: 0.6918
- Recall: 0.7466

## Diagnosis Vector
- small_object_score: 0.2943
- low_contrast_score: 0.5093
- class_imbalance_score: 0.9853
- localization_score: 0.0090
- false_positive_score: 0.3082

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=38 fp=6 fn=0 precision=0.8636 recall=1.0000
- OK3: gt=102 tp=100 fp=25 fn=2 precision=0.8000 recall=0.9804
- 加强筋打伤: gt=6 tp=4 fp=0 fn=2 precision=1.0000 recall=0.6667
- 开裂: gt=3 tp=0 fp=0 fn=3 precision=0.0000 recall=0.0000
- 油污: gt=17 tp=3 fp=28 fn=14 precision=0.0968 recall=0.1765
- 浅划伤: gt=14 tp=0 fp=0 fn=14 precision=0.0000 recall=0.0000
- 漏背锡: gt=32 tp=24 fp=7 fn=8 precision=0.7742 recall=0.7500
- 碰伤: gt=133 tp=99 fp=22 fn=33 precision=0.8182 recall=0.7444
- 脏污: gt=28 tp=10 fp=9 fn=18 precision=0.5263 recall=0.3571
- 轮廓划伤: gt=30 tp=26 fp=22 fn=3 precision=0.5417 recall=0.8667
- 锡丝残留: gt=6 tp=0 fp=0 fn=6 precision=0.0000 recall=0.0000
- 锡尖: gt=12 tp=11 fp=9 fn=0 precision=0.5500 recall=0.9167
- 锡膏: gt=21 tp=15 fp=19 fn=5 precision=0.4412 recall=0.7143
