# Validation Error Diagnosis

- Status: completed
- TP: 371
- FP: 188
- FN: 533
- Precision: 0.6637
- Recall: 0.4099

## Diagnosis Vector
- small_object_score: 0.7385
- low_contrast_score: 0.5042
- class_imbalance_score: 0.9953
- localization_score: 0.0011
- false_positive_score: 0.3363

## Issues
- small_object_low_recall severity=high suggestions=tiling, object-aware-crop, scale, copy-paste, mild-geometry
- background_interference severity=medium suggestions=background-diversity, light-noise, illumination-jitter
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=88 fn=0 precision=0.4211 recall=1.0000
- OK3: gt=274 tp=263 fp=78 fn=10 precision=0.7713 recall=0.9599
- 加强筋打伤: gt=8 tp=0 fp=0 fn=8 precision=0.0000 recall=0.0000
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=0 fp=0 fn=18 precision=0.0000 recall=0.0000
- 浅划伤: gt=13 tp=0 fp=0 fn=13 precision=0.0000 recall=0.0000
- 漏背锡: gt=46 tp=0 fp=5 fn=46 precision=0.0000 recall=0.0000
- 碰伤: gt=269 tp=44 fp=17 fn=225 precision=0.7213 recall=0.1636
- 脏污: gt=42 tp=0 fp=0 fn=42 precision=0.0000 recall=0.0000
- 轮廓划伤: gt=84 tp=0 fp=0 fn=84 precision=0.0000 recall=0.0000
- 锡丝残留: gt=12 tp=0 fp=0 fn=12 precision=0.0000 recall=0.0000
- 锡尖: gt=27 tp=0 fp=0 fn=27 precision=0.0000 recall=0.0000
- 锡膏: gt=46 tp=0 fp=0 fn=46 precision=0.0000 recall=0.0000
