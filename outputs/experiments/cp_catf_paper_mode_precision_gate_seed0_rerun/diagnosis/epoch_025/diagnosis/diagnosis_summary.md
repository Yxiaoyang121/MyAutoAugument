# Validation Error Diagnosis

- Status: completed
- TP: 361
- FP: 154
- FN: 73
- Precision: 0.7010
- Recall: 0.8167

## Diagnosis Vector
- small_object_score: 0.2174
- low_contrast_score: 0.5904
- class_imbalance_score: 0.9236
- localization_score: 0.0181
- false_positive_score: 0.2990

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=38 fp=2 fn=0 precision=0.9500 recall=1.0000
- OK3: gt=102 tp=101 fp=24 fn=1 precision=0.8080 recall=0.9902
- 加强筋打伤: gt=6 tp=6 fp=0 fn=0 precision=1.0000 recall=1.0000
- 开裂: gt=3 tp=3 fp=2 fn=0 precision=0.6000 recall=1.0000
- 油污: gt=17 tp=3 fp=6 fn=14 precision=0.3333 recall=0.1765
- 浅划伤: gt=14 tp=7 fp=9 fn=5 precision=0.4375 recall=0.5000
- 漏背锡: gt=32 tp=26 fp=12 fn=5 precision=0.6842 recall=0.8125
- 碰伤: gt=133 tp=102 fp=23 fn=31 precision=0.8160 recall=0.7669
- 脏污: gt=28 tp=13 fp=45 fn=10 precision=0.2241 recall=0.4643
- 轮廓划伤: gt=30 tp=29 fp=9 fn=1 precision=0.7632 recall=0.9667
- 锡丝残留: gt=6 tp=3 fp=1 fn=3 precision=0.7500 recall=0.5000
- 锡尖: gt=12 tp=12 fp=15 fn=0 precision=0.4444 recall=1.0000
- 锡膏: gt=21 tp=18 fp=6 fn=3 precision=0.7500 recall=0.8571
