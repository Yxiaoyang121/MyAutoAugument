# Validation Error Diagnosis

- Status: completed
- TP: 699
- FP: 300
- FN: 194
- Precision: 0.6997
- Recall: 0.7724

## Diagnosis Vector
- small_object_score: 0.2852
- low_contrast_score: 0.5322
- class_imbalance_score: 0.9953
- localization_score: 0.0133
- false_positive_score: 0.3003

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=22 fn=0 precision=0.7442 recall=1.0000
- OK3: gt=274 tp=274 fp=72 fn=0 precision=0.7919 recall=1.0000
- 加强筋打伤: gt=8 tp=3 fp=1 fn=5 precision=0.7500 recall=0.3750
- 开裂: gt=2 tp=0 fp=2 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=4 fp=10 fn=13 precision=0.2857 recall=0.2222
- 浅划伤: gt=13 tp=8 fp=6 fn=5 precision=0.5714 recall=0.6154
- 漏背锡: gt=46 tp=33 fp=30 fn=12 precision=0.5238 recall=0.7174
- 碰伤: gt=269 tp=178 fp=86 fn=87 precision=0.6742 recall=0.6617
- 脏污: gt=42 tp=23 fp=14 fn=19 precision=0.6216 recall=0.5476
- 轮廓划伤: gt=84 tp=48 fp=31 fn=30 precision=0.6076 recall=0.5714
- 锡丝残留: gt=12 tp=9 fp=13 fn=3 precision=0.4091 recall=0.7500
- 锡尖: gt=27 tp=25 fp=5 fn=2 precision=0.8333 recall=0.9259
- 锡膏: gt=46 tp=30 fp=8 fn=16 precision=0.7895 recall=0.6522
