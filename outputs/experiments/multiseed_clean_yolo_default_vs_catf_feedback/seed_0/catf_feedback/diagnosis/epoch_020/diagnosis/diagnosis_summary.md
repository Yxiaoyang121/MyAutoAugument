# Validation Error Diagnosis

- Status: completed
- TP: 672
- FP: 338
- FN: 215
- Precision: 0.6653
- Recall: 0.7425

## Diagnosis Vector
- small_object_score: 0.3396
- low_contrast_score: 0.5607
- class_imbalance_score: 0.9953
- localization_score: 0.0199
- false_positive_score: 0.3347

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=24 fn=0 precision=0.7273 recall=1.0000
- OK3: gt=274 tp=274 fp=71 fn=0 precision=0.7942 recall=1.0000
- 加强筋打伤: gt=8 tp=6 fp=0 fn=2 precision=1.0000 recall=0.7500
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=5 fp=40 fn=13 precision=0.1111 recall=0.2778
- 浅划伤: gt=13 tp=6 fp=8 fn=7 precision=0.4286 recall=0.4615
- 漏背锡: gt=46 tp=30 fp=28 fn=11 precision=0.5172 recall=0.6522
- 碰伤: gt=269 tp=155 fp=70 fn=109 precision=0.6889 recall=0.5762
- 脏污: gt=42 tp=18 fp=13 fn=24 precision=0.5806 recall=0.4286
- 轮廓划伤: gt=84 tp=45 fp=27 fn=31 precision=0.6250 recall=0.5357
- 锡丝残留: gt=12 tp=5 fp=9 fn=7 precision=0.3571 recall=0.4167
- 锡尖: gt=27 tp=25 fp=8 fn=2 precision=0.7576 recall=0.9259
- 锡膏: gt=46 tp=39 fp=40 fn=7 precision=0.4937 recall=0.8478
