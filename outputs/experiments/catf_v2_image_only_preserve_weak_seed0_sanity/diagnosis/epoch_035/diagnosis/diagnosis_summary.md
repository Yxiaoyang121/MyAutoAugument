# Validation Error Diagnosis

- Status: completed
- TP: 724
- FP: 284
- FN: 163
- Precision: 0.7183
- Recall: 0.8000

## Diagnosis Vector
- small_object_score: 0.2547
- low_contrast_score: 0.5255
- class_imbalance_score: 0.8640
- localization_score: 0.0199
- false_positive_score: 0.2817

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=21 fn=0 precision=0.7529 recall=1.0000
- OK3: gt=274 tp=274 fp=69 fn=0 precision=0.7988 recall=1.0000
- 加强筋打伤: gt=8 tp=3 fp=1 fn=5 precision=0.7500 recall=0.3750
- 开裂: gt=2 tp=1 fp=2 fn=1 precision=0.3333 recall=0.5000
- 油污: gt=18 tp=7 fp=19 fn=10 precision=0.2692 recall=0.3889
- 浅划伤: gt=13 tp=10 fp=9 fn=3 precision=0.5263 recall=0.7692
- 漏背锡: gt=46 tp=32 fp=17 fn=10 precision=0.6531 recall=0.6957
- 碰伤: gt=269 tp=193 fp=64 fn=71 precision=0.7510 recall=0.7175
- 脏污: gt=42 tp=18 fp=22 fn=22 precision=0.4500 recall=0.4286
- 轮廓划伤: gt=84 tp=54 fp=34 fn=25 precision=0.6136 recall=0.6429
- 锡丝残留: gt=12 tp=9 fp=7 fn=3 precision=0.5625 recall=0.7500
- 锡尖: gt=27 tp=27 fp=5 fn=0 precision=0.8438 recall=1.0000
- 锡膏: gt=46 tp=32 fp=14 fn=13 precision=0.6957 recall=0.6957
