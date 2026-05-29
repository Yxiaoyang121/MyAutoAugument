# Validation Error Diagnosis

- Status: completed
- TP: 645
- FP: 547
- FN: 239
- Precision: 0.5411
- Recall: 0.7127

## Diagnosis Vector
- small_object_score: 0.3616
- low_contrast_score: 0.5249
- class_imbalance_score: 0.9953
- localization_score: 0.0232
- false_positive_score: 0.4589

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=29 fn=0 precision=0.6882 recall=1.0000
- OK3: gt=274 tp=273 fp=71 fn=1 precision=0.7936 recall=0.9964
- 加强筋打伤: gt=8 tp=0 fp=0 fn=8 precision=0.0000 recall=0.0000
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=6 fp=71 fn=12 precision=0.0779 recall=0.3333
- 浅划伤: gt=13 tp=5 fp=5 fn=6 precision=0.5000 recall=0.3846
- 漏背锡: gt=46 tp=18 fp=28 fn=23 precision=0.3913 recall=0.3913
- 碰伤: gt=269 tp=165 fp=94 fn=98 precision=0.6371 recall=0.6134
- 脏污: gt=42 tp=25 fp=141 fn=13 precision=0.1506 recall=0.5952
- 轮廓划伤: gt=84 tp=32 fp=25 fn=52 precision=0.5614 recall=0.3810
- 锡丝残留: gt=12 tp=1 fp=0 fn=11 precision=1.0000 recall=0.0833
- 锡尖: gt=27 tp=18 fp=0 fn=7 precision=1.0000 recall=0.6667
- 锡膏: gt=46 tp=38 fp=83 fn=6 precision=0.3140 recall=0.8261
