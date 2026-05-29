# Validation Error Diagnosis

- Status: completed
- TP: 610
- FP: 346
- FN: 277
- Precision: 0.6381
- Recall: 0.6740

## Diagnosis Vector
- small_object_score: 0.4092
- low_contrast_score: 0.5374
- class_imbalance_score: 0.9953
- localization_score: 0.0199
- false_positive_score: 0.3619

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=26 fn=0 precision=0.7111 recall=1.0000
- OK3: gt=274 tp=274 fp=66 fn=0 precision=0.8059 recall=1.0000
- 加强筋打伤: gt=8 tp=0 fp=0 fn=8 precision=0.0000 recall=0.0000
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=5 fp=32 fn=13 precision=0.1351 recall=0.2778
- 浅划伤: gt=13 tp=2 fp=0 fn=11 precision=1.0000 recall=0.1538
- 漏背锡: gt=46 tp=26 fp=27 fn=16 precision=0.4906 recall=0.5652
- 碰伤: gt=269 tp=136 fp=45 fn=128 precision=0.7514 recall=0.5056
- 脏污: gt=42 tp=16 fp=76 fn=26 precision=0.1739 recall=0.3810
- 轮廓划伤: gt=84 tp=32 fp=39 fn=43 precision=0.4507 recall=0.3810
- 锡丝残留: gt=12 tp=0 fp=0 fn=12 precision=0.0000 recall=0.0000
- 锡尖: gt=27 tp=25 fp=17 fn=2 precision=0.5952 recall=0.9259
- 锡膏: gt=46 tp=30 fp=18 fn=16 precision=0.6250 recall=0.6522
