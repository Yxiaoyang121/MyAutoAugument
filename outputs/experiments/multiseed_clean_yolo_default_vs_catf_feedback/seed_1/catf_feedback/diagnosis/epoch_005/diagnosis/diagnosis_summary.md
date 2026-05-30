# Validation Error Diagnosis

- Status: completed
- TP: 551
- FP: 233
- FN: 331
- Precision: 0.7028
- Recall: 0.6088

## Diagnosis Vector
- small_object_score: 0.4686
- low_contrast_score: 0.5051
- class_imbalance_score: 0.9953
- localization_score: 0.0254
- false_positive_score: 0.2972

## Issues
- small_object_low_recall severity=medium suggestions=tiling, object-aware-crop, scale, copy-paste, mild-geometry
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=21 fn=0 precision=0.7529 recall=1.0000
- OK3: gt=274 tp=268 fp=72 fn=4 precision=0.7882 recall=0.9781
- 加强筋打伤: gt=8 tp=0 fp=0 fn=8 precision=0.0000 recall=0.0000
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=0 fp=0 fn=18 precision=0.0000 recall=0.0000
- 浅划伤: gt=13 tp=0 fp=0 fn=13 precision=0.0000 recall=0.0000
- 漏背锡: gt=46 tp=9 fp=14 fn=36 precision=0.3913 recall=0.1957
- 碰伤: gt=269 tp=127 fp=53 fn=134 precision=0.7056 recall=0.4721
- 脏污: gt=42 tp=7 fp=8 fn=35 precision=0.4667 recall=0.1667
- 轮廓划伤: gt=84 tp=27 fp=12 fn=52 precision=0.6923 recall=0.3214
- 锡丝残留: gt=12 tp=0 fp=0 fn=12 precision=0.0000 recall=0.0000
- 锡尖: gt=27 tp=18 fp=8 fn=6 precision=0.6923 recall=0.6667
- 锡膏: gt=46 tp=31 fp=45 fn=11 precision=0.4079 recall=0.6739
