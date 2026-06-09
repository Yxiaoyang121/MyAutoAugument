# Validation Error Diagnosis

- Status: completed
- TP: 312
- FP: 182
- FN: 126
- Precision: 0.6316
- Recall: 0.7059

## Diagnosis Vector
- small_object_score: 0.3211
- low_contrast_score: 0.4734
- class_imbalance_score: 0.9853
- localization_score: 0.0090
- false_positive_score: 0.3684

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=38 fp=4 fn=0 precision=0.9048 recall=1.0000
- OK3: gt=102 tp=100 fp=27 fn=1 precision=0.7874 recall=0.9804
- 加强筋打伤: gt=6 tp=4 fp=1 fn=2 precision=0.8000 recall=0.6667
- 开裂: gt=3 tp=0 fp=0 fn=3 precision=0.0000 recall=0.0000
- 油污: gt=17 tp=1 fp=23 fn=16 precision=0.0417 recall=0.0588
- 浅划伤: gt=14 tp=0 fp=5 fn=14 precision=0.0000 recall=0.0000
- 漏背锡: gt=32 tp=21 fp=22 fn=9 precision=0.4884 recall=0.6562
- 碰伤: gt=133 tp=91 fp=27 fn=41 precision=0.7712 recall=0.6842
- 脏污: gt=28 tp=5 fp=3 fn=23 precision=0.6250 recall=0.1786
- 轮廓划伤: gt=30 tp=28 fp=31 fn=2 precision=0.4746 recall=0.9333
- 锡丝残留: gt=6 tp=0 fp=0 fn=6 precision=0.0000 recall=0.0000
- 锡尖: gt=12 tp=12 fp=23 fn=0 precision=0.3429 recall=1.0000
- 锡膏: gt=21 tp=12 fp=16 fn=9 precision=0.4286 recall=0.5714
