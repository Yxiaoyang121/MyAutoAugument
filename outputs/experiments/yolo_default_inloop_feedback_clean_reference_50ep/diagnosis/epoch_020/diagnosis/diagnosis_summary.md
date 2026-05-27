# Validation Error Diagnosis

- Status: completed
- TP: 694
- FP: 339
- FN: 195
- Precision: 0.6718
- Recall: 0.7669

## Diagnosis Vector
- small_object_score: 0.2852
- low_contrast_score: 0.5241
- class_imbalance_score: 0.9953
- localization_score: 0.0177
- false_positive_score: 0.3282

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=22 fn=0 precision=0.7442 recall=1.0000
- OK3: gt=274 tp=274 fp=70 fn=0 precision=0.7965 recall=1.0000
- 加强筋打伤: gt=8 tp=6 fp=0 fn=2 precision=1.0000 recall=0.7500
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=5 fp=26 fn=13 precision=0.1613 recall=0.2778
- 浅划伤: gt=13 tp=6 fp=5 fn=7 precision=0.5455 recall=0.4615
- 漏背锡: gt=46 tp=30 fp=18 fn=14 precision=0.6250 recall=0.6522
- 碰伤: gt=269 tp=175 fp=81 fn=89 precision=0.6836 recall=0.6506
- 脏污: gt=42 tp=28 fp=38 fn=11 precision=0.4242 recall=0.6667
- 轮廓划伤: gt=84 tp=46 fp=28 fn=33 precision=0.6216 recall=0.5476
- 锡丝残留: gt=12 tp=9 fp=11 fn=3 precision=0.4500 recall=0.7500
- 锡尖: gt=27 tp=24 fp=13 fn=3 precision=0.6486 recall=0.8889
- 锡膏: gt=46 tp=27 fp=27 fn=18 precision=0.5000 recall=0.5870
