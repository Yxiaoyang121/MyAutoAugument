# Validation Error Diagnosis

- Status: completed
- TP: 735
- FP: 391
- FN: 154
- Precision: 0.6528
- Recall: 0.8122

## Diagnosis Vector
- small_object_score: 0.2479
- low_contrast_score: 0.5461
- class_imbalance_score: 0.9953
- localization_score: 0.0177
- false_positive_score: 0.3472

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=20 fn=0 precision=0.7619 recall=1.0000
- OK3: gt=274 tp=274 fp=69 fn=0 precision=0.7988 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=0 fn=1 precision=1.0000 recall=0.8750
- 开裂: gt=2 tp=0 fp=1 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=11 fp=54 fn=6 precision=0.1692 recall=0.6111
- 浅划伤: gt=13 tp=6 fp=0 fn=7 precision=1.0000 recall=0.4615
- 漏背锡: gt=46 tp=36 fp=42 fn=8 precision=0.4615 recall=0.7826
- 碰伤: gt=269 tp=191 fp=75 fn=71 precision=0.7180 recall=0.7100
- 脏污: gt=42 tp=24 fp=38 fn=14 precision=0.3871 recall=0.5714
- 轮廓划伤: gt=84 tp=49 fp=54 fn=33 precision=0.4757 recall=0.5833
- 锡丝残留: gt=12 tp=9 fp=9 fn=3 precision=0.5000 recall=0.7500
- 锡尖: gt=27 tp=24 fp=5 fn=3 precision=0.8276 recall=0.8889
- 锡膏: gt=46 tp=40 fp=24 fn=6 precision=0.6250 recall=0.8696
