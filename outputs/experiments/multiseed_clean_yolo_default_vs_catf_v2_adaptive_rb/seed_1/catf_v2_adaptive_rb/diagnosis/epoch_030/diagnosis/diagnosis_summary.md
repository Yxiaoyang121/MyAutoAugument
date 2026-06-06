# Validation Error Diagnosis

- Status: completed
- TP: 701
- FP: 377
- FN: 179
- Precision: 0.6503
- Recall: 0.7746

## Diagnosis Vector
- small_object_score: 0.2903
- low_contrast_score: 0.5397
- class_imbalance_score: 0.8494
- localization_score: 0.0276
- false_positive_score: 0.3497

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=22 fn=0 precision=0.7442 recall=1.0000
- OK3: gt=274 tp=274 fp=73 fn=0 precision=0.7896 recall=1.0000
- 加强筋打伤: gt=8 tp=8 fp=11 fn=0 precision=0.4211 recall=1.0000
- 开裂: gt=2 tp=2 fp=9 fn=0 precision=0.1818 recall=1.0000
- 油污: gt=18 tp=9 fp=35 fn=9 precision=0.2045 recall=0.5000
- 浅划伤: gt=13 tp=6 fp=5 fn=5 precision=0.5455 recall=0.4615
- 漏背锡: gt=46 tp=30 fp=39 fn=10 precision=0.4348 recall=0.6522
- 碰伤: gt=269 tp=178 fp=97 fn=86 precision=0.6473 recall=0.6617
- 脏污: gt=42 tp=24 fp=22 fn=14 precision=0.5217 recall=0.5714
- 轮廓划伤: gt=84 tp=45 fp=41 fn=31 precision=0.5233 recall=0.5357
- 锡丝残留: gt=12 tp=5 fp=5 fn=7 precision=0.5000 recall=0.4167
- 锡尖: gt=27 tp=23 fp=4 fn=4 precision=0.8519 recall=0.8519
- 锡膏: gt=46 tp=33 fp=14 fn=13 precision=0.7021 recall=0.7174
