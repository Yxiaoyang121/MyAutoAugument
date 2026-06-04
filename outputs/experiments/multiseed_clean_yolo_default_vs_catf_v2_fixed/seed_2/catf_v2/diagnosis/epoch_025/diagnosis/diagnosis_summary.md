# Validation Error Diagnosis

- Status: completed
- TP: 711
- FP: 396
- FN: 171
- Precision: 0.6423
- Recall: 0.7856

## Diagnosis Vector
- small_object_score: 0.2869
- low_contrast_score: 0.5798
- class_imbalance_score: 0.9175
- localization_score: 0.0254
- false_positive_score: 0.3577

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=25 fn=0 precision=0.7191 recall=1.0000
- OK3: gt=274 tp=272 fp=69 fn=2 precision=0.7977 recall=0.9927
- 加强筋打伤: gt=8 tp=5 fp=0 fn=3 precision=1.0000 recall=0.6250
- 开裂: gt=2 tp=2 fp=0 fn=0 precision=1.0000 recall=1.0000
- 油污: gt=18 tp=4 fp=20 fn=14 precision=0.1667 recall=0.2222
- 浅划伤: gt=13 tp=10 fp=12 fn=3 precision=0.4545 recall=0.7692
- 漏背锡: gt=46 tp=36 fp=60 fn=8 precision=0.3750 recall=0.7826
- 碰伤: gt=269 tp=172 fp=100 fn=90 precision=0.6324 recall=0.6394
- 脏污: gt=42 tp=25 fp=28 fn=14 precision=0.4717 recall=0.5952
- 轮廓划伤: gt=84 tp=56 fp=36 fn=19 precision=0.6087 recall=0.6667
- 锡丝残留: gt=12 tp=8 fp=3 fn=4 precision=0.7273 recall=0.6667
- 锡尖: gt=27 tp=19 fp=3 fn=8 precision=0.8636 recall=0.7037
- 锡膏: gt=46 tp=38 fp=40 fn=6 precision=0.4872 recall=0.8261
