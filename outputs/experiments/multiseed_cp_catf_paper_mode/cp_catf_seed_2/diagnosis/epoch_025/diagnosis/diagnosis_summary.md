# Validation Error Diagnosis

- Status: completed
- TP: 365
- FP: 181
- FN: 68
- Precision: 0.6685
- Recall: 0.8258

## Diagnosis Vector
- small_object_score: 0.1973
- low_contrast_score: 0.5735
- class_imbalance_score: 0.9236
- localization_score: 0.0204
- false_positive_score: 0.3315

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=38 fp=3 fn=0 precision=0.9268 recall=1.0000
- OK3: gt=102 tp=101 fp=24 fn=1 precision=0.8080 recall=0.9902
- 加强筋打伤: gt=6 tp=6 fp=2 fn=0 precision=0.7500 recall=1.0000
- 开裂: gt=3 tp=2 fp=0 fn=1 precision=1.0000 recall=0.6667
- 油污: gt=17 tp=3 fp=10 fn=14 precision=0.2308 recall=0.1765
- 浅划伤: gt=14 tp=3 fp=8 fn=9 precision=0.2727 recall=0.2143
- 漏背锡: gt=32 tp=25 fp=19 fn=5 precision=0.5682 recall=0.7812
- 碰伤: gt=133 tp=104 fp=38 fn=28 precision=0.7324 recall=0.7820
- 脏污: gt=28 tp=17 fp=48 fn=9 precision=0.2615 recall=0.6071
- 轮廓划伤: gt=30 tp=30 fp=7 fn=0 precision=0.8108 recall=1.0000
- 锡丝残留: gt=6 tp=6 fp=3 fn=0 precision=0.6667 recall=1.0000
- 锡尖: gt=12 tp=12 fp=15 fn=0 precision=0.4444 recall=1.0000
- 锡膏: gt=21 tp=18 fp=4 fn=1 precision=0.8182 recall=0.8571
