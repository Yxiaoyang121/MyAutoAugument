# Validation Error Diagnosis

- Status: completed
- TP: 710
- FP: 387
- FN: 170
- Precision: 0.6472
- Recall: 0.7845

## Diagnosis Vector
- small_object_score: 0.2699
- low_contrast_score: 0.5859
- class_imbalance_score: 0.8786
- localization_score: 0.0276
- false_positive_score: 0.3528

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=21 fn=0 precision=0.7529 recall=1.0000
- OK3: gt=274 tp=274 fp=67 fn=0 precision=0.8035 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=1 fn=1 precision=0.8750 recall=0.8750
- 开裂: gt=2 tp=2 fp=28 fn=0 precision=0.0667 recall=1.0000
- 油污: gt=18 tp=6 fp=25 fn=12 precision=0.1935 recall=0.3333
- 浅划伤: gt=13 tp=6 fp=3 fn=7 precision=0.6667 recall=0.4615
- 漏背锡: gt=46 tp=32 fp=64 fn=5 precision=0.3333 recall=0.6957
- 碰伤: gt=269 tp=188 fp=75 fn=75 precision=0.7148 recall=0.6989
- 脏污: gt=42 tp=21 fp=30 fn=15 precision=0.4118 recall=0.5000
- 轮廓划伤: gt=84 tp=39 fp=20 fn=43 precision=0.6610 recall=0.4643
- 锡丝残留: gt=12 tp=9 fp=10 fn=3 precision=0.4737 recall=0.7500
- 锡尖: gt=27 tp=25 fp=9 fn=2 precision=0.7353 recall=0.9259
- 锡膏: gt=46 tp=37 fp=34 fn=7 precision=0.5211 recall=0.8043
