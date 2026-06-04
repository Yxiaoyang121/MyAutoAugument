# Validation Error Diagnosis

- Status: completed
- TP: 699
- FP: 317
- FN: 188
- Precision: 0.6880
- Recall: 0.7724

## Diagnosis Vector
- small_object_score: 0.2784
- low_contrast_score: 0.5149
- class_imbalance_score: 0.9953
- localization_score: 0.0199
- false_positive_score: 0.3120

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=21 fn=0 precision=0.7529 recall=1.0000
- OK3: gt=274 tp=274 fp=67 fn=0 precision=0.8035 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=3 fn=1 precision=0.7000 recall=0.8750
- 开裂: gt=2 tp=0 fp=1 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=4 fp=19 fn=14 precision=0.1739 recall=0.2222
- 浅划伤: gt=13 tp=5 fp=1 fn=8 precision=0.8333 recall=0.3846
- 漏背锡: gt=46 tp=36 fp=54 fn=8 precision=0.4000 recall=0.7826
- 碰伤: gt=269 tp=175 fp=75 fn=88 precision=0.7000 recall=0.6506
- 脏污: gt=42 tp=21 fp=21 fn=16 precision=0.5000 recall=0.5000
- 轮廓划伤: gt=84 tp=53 fp=35 fn=27 precision=0.6023 recall=0.6310
- 锡丝残留: gt=12 tp=7 fp=4 fn=5 precision=0.6364 recall=0.5833
- 锡尖: gt=27 tp=23 fp=3 fn=3 precision=0.8846 recall=0.8519
- 锡膏: gt=46 tp=30 fp=13 fn=16 precision=0.6977 recall=0.6522
