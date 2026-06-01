# Validation Error Diagnosis

- Status: completed
- TP: 528
- FP: 179
- FN: 369
- Precision: 0.7468
- Recall: 0.5834

## Diagnosis Vector
- small_object_score: 0.5059
- low_contrast_score: 0.5072
- class_imbalance_score: 0.9953
- localization_score: 0.0088
- false_positive_score: 0.2532

## Issues
- small_object_low_recall severity=medium suggestions=tiling, object-aware-crop, scale, copy-paste, mild-geometry
- background_interference severity=medium suggestions=background-diversity, light-noise, illumination-jitter
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=17 fn=0 precision=0.7901 recall=1.0000
- OK3: gt=274 tp=269 fp=44 fn=4 precision=0.8594 recall=0.9818
- 加强筋打伤: gt=8 tp=0 fp=0 fn=8 precision=0.0000 recall=0.0000
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=0 fp=0 fn=18 precision=0.0000 recall=0.0000
- 浅划伤: gt=13 tp=2 fp=0 fn=11 precision=1.0000 recall=0.1538
- 漏背锡: gt=46 tp=17 fp=8 fn=24 precision=0.6800 recall=0.3696
- 碰伤: gt=269 tp=104 fp=40 fn=164 precision=0.7222 recall=0.3866
- 脏污: gt=42 tp=0 fp=0 fn=42 precision=0.0000 recall=0.0000
- 轮廓划伤: gt=84 tp=35 fp=59 fn=49 precision=0.3723 recall=0.4167
- 锡丝残留: gt=12 tp=0 fp=0 fn=12 precision=0.0000 recall=0.0000
- 锡尖: gt=27 tp=12 fp=0 fn=15 precision=1.0000 recall=0.4444
- 锡膏: gt=46 tp=25 fp=11 fn=20 precision=0.6944 recall=0.5435
