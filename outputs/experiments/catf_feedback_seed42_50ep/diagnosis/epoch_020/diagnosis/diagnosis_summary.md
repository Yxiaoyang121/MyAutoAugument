# Validation Error Diagnosis

- Status: completed
- TP: 680
- FP: 345
- FN: 216
- Precision: 0.6634
- Recall: 0.7514

## Diagnosis Vector
- small_object_score: 0.3124
- low_contrast_score: 0.5273
- class_imbalance_score: 0.9515
- localization_score: 0.0099
- false_positive_score: 0.3366

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=21 fn=0 precision=0.7529 recall=1.0000
- OK3: gt=274 tp=272 fp=70 fn=2 precision=0.7953 recall=0.9927
- 加强筋打伤: gt=8 tp=1 fp=0 fn=7 precision=1.0000 recall=0.1250
- 开裂: gt=2 tp=1 fp=18 fn=1 precision=0.0526 recall=0.5000
- 油污: gt=18 tp=8 fp=45 fn=10 precision=0.1509 recall=0.4444
- 浅划伤: gt=13 tp=5 fp=1 fn=7 precision=0.8333 recall=0.3846
- 漏背锡: gt=46 tp=29 fp=16 fn=16 precision=0.6444 recall=0.6304
- 碰伤: gt=269 tp=179 fp=67 fn=85 precision=0.7276 recall=0.6654
- 脏污: gt=42 tp=24 fp=50 fn=18 precision=0.3243 recall=0.5714
- 轮廓划伤: gt=84 tp=43 fp=34 fn=39 precision=0.5584 recall=0.5119
- 锡丝残留: gt=12 tp=3 fp=2 fn=9 precision=0.6000 recall=0.2500
- 锡尖: gt=27 tp=25 fp=9 fn=2 precision=0.7353 recall=0.9259
- 锡膏: gt=46 tp=26 fp=12 fn=20 precision=0.6842 recall=0.5652
