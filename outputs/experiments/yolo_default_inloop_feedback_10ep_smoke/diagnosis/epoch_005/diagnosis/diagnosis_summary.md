# Validation Error Diagnosis

- Status: completed
- TP: 573
- FP: 250
- FN: 324
- Precision: 0.6962
- Recall: 0.6331

## Diagnosis Vector
- small_object_score: 0.4295
- low_contrast_score: 0.4881
- class_imbalance_score: 0.9953
- localization_score: 0.0088
- false_positive_score: 0.3038

## Issues
- small_object_low_recall severity=medium suggestions=tiling, object-aware-crop, scale, copy-paste, mild-geometry
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=13 fn=0 precision=0.8312 recall=1.0000
- OK3: gt=274 tp=271 fp=49 fn=2 precision=0.8469 recall=0.9891
- 加强筋打伤: gt=8 tp=0 fp=0 fn=8 precision=0.0000 recall=0.0000
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=0 fp=0 fn=18 precision=0.0000 recall=0.0000
- 浅划伤: gt=13 tp=0 fp=0 fn=13 precision=0.0000 recall=0.0000
- 漏背锡: gt=46 tp=13 fp=7 fn=32 precision=0.6500 recall=0.2826
- 碰伤: gt=269 tp=123 fp=65 fn=144 precision=0.6543 recall=0.4572
- 脏污: gt=42 tp=0 fp=0 fn=42 precision=0.0000 recall=0.0000
- 轮廓划伤: gt=84 tp=59 fp=93 fn=22 precision=0.3882 recall=0.7024
- 锡丝残留: gt=12 tp=0 fp=0 fn=12 precision=0.0000 recall=0.0000
- 锡尖: gt=27 tp=19 fp=2 fn=7 precision=0.9048 recall=0.7037
- 锡膏: gt=46 tp=24 fp=21 fn=22 precision=0.5333 recall=0.5217
