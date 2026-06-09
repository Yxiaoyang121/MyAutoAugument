# Validation Error Diagnosis

- Status: completed
- TP: 270
- FP: 106
- FN: 163
- Precision: 0.7181
- Recall: 0.6109

## Diagnosis Vector
- small_object_score: 0.4181
- low_contrast_score: 0.4433
- class_imbalance_score: 0.9853
- localization_score: 0.0204
- false_positive_score: 0.2819

## Issues
- small_object_low_recall severity=medium suggestions=tiling, object-aware-crop, scale, copy-paste, mild-geometry
- background_interference severity=medium suggestions=background-diversity, light-noise, illumination-jitter
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=38 fp=0 fn=0 precision=1.0000 recall=1.0000
- OK3: gt=102 tp=97 fp=15 fn=5 precision=0.8661 recall=0.9510
- 加强筋打伤: gt=6 tp=0 fp=0 fn=6 precision=0.0000 recall=0.0000
- 开裂: gt=3 tp=0 fp=0 fn=3 precision=0.0000 recall=0.0000
- 油污: gt=17 tp=0 fp=0 fn=17 precision=0.0000 recall=0.0000
- 浅划伤: gt=14 tp=0 fp=0 fn=14 precision=0.0000 recall=0.0000
- 漏背锡: gt=32 tp=15 fp=26 fn=12 precision=0.3659 recall=0.4688
- 碰伤: gt=133 tp=93 fp=49 fn=39 precision=0.6549 recall=0.6992
- 脏污: gt=28 tp=0 fp=0 fn=28 precision=0.0000 recall=0.0000
- 轮廓划伤: gt=30 tp=16 fp=4 fn=12 precision=0.8000 recall=0.5333
- 锡丝残留: gt=6 tp=0 fp=0 fn=6 precision=0.0000 recall=0.0000
- 锡尖: gt=12 tp=11 fp=12 fn=0 precision=0.4783 recall=0.9167
- 锡膏: gt=21 tp=0 fp=0 fn=21 precision=0.0000 recall=0.0000
