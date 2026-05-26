# Validation Error Diagnosis

- Status: completed
- TP: 496
- FP: 209
- FN: 395
- Precision: 0.7035
- Recall: 0.5481

## Diagnosis Vector
- small_object_score: 0.5331
- low_contrast_score: 0.4571
- class_imbalance_score: 0.9953
- localization_score: 0.0155
- false_positive_score: 0.2965

## Issues
- small_object_low_recall severity=medium suggestions=tiling, object-aware-crop, scale, copy-paste, mild-geometry
- background_interference severity=medium suggestions=background-diversity, light-noise, illumination-jitter
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=38 fn=0 precision=0.6275 recall=1.0000
- OK3: gt=274 tp=265 fp=53 fn=8 precision=0.8333 recall=0.9672
- 加强筋打伤: gt=8 tp=0 fp=0 fn=8 precision=0.0000 recall=0.0000
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=0 fp=0 fn=18 precision=0.0000 recall=0.0000
- 浅划伤: gt=13 tp=0 fp=0 fn=13 precision=0.0000 recall=0.0000
- 漏背锡: gt=46 tp=7 fp=17 fn=39 precision=0.2917 recall=0.1522
- 碰伤: gt=269 tp=109 fp=73 fn=154 precision=0.5989 recall=0.4052
- 脏污: gt=42 tp=3 fp=4 fn=39 precision=0.4286 recall=0.0714
- 轮廓划伤: gt=84 tp=35 fp=19 fn=42 precision=0.6481 recall=0.4167
- 锡丝残留: gt=12 tp=0 fp=0 fn=12 precision=0.0000 recall=0.0000
- 锡尖: gt=27 tp=6 fp=0 fn=21 precision=1.0000 recall=0.2222
- 锡膏: gt=46 tp=7 fp=5 fn=39 precision=0.5833 recall=0.1522
