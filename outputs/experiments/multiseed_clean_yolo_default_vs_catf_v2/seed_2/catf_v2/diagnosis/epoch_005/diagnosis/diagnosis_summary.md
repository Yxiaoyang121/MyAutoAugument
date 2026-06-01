# Validation Error Diagnosis

- Status: completed
- TP: 520
- FP: 346
- FN: 354
- Precision: 0.6005
- Recall: 0.5746

## Diagnosis Vector
- small_object_score: 0.5229
- low_contrast_score: 0.4855
- class_imbalance_score: 0.9953
- localization_score: 0.0343
- false_positive_score: 0.3995

## Issues
- small_object_low_recall severity=medium suggestions=tiling, object-aware-crop, scale, copy-paste, mild-geometry
- background_interference severity=medium suggestions=background-diversity, light-noise, illumination-jitter
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=50 fn=0 precision=0.5614 recall=1.0000
- OK3: gt=274 tp=271 fp=71 fn=3 precision=0.7924 recall=0.9891
- 加强筋打伤: gt=8 tp=0 fp=0 fn=8 precision=0.0000 recall=0.0000
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=0 fp=0 fn=18 precision=0.0000 recall=0.0000
- 浅划伤: gt=13 tp=0 fp=0 fn=13 precision=0.0000 recall=0.0000
- 漏背锡: gt=46 tp=20 fp=16 fn=22 precision=0.5556 recall=0.4348
- 碰伤: gt=269 tp=120 fp=66 fn=143 precision=0.6452 recall=0.4461
- 脏污: gt=42 tp=0 fp=8 fn=42 precision=0.0000 recall=0.0000
- 轮廓划伤: gt=84 tp=19 fp=56 fn=48 precision=0.2533 recall=0.2262
- 锡丝残留: gt=12 tp=0 fp=0 fn=12 precision=0.0000 recall=0.0000
- 锡尖: gt=27 tp=25 fp=76 fn=0 precision=0.2475 recall=0.9259
- 锡膏: gt=46 tp=1 fp=3 fn=43 precision=0.2500 recall=0.0217
