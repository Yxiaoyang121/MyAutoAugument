# Validation Error Diagnosis

- Status: completed
- TP: 659
- FP: 396
- FN: 228
- Precision: 0.6246
- Recall: 0.7282

## Diagnosis Vector
- small_object_score: 0.3294
- low_contrast_score: 0.4673
- class_imbalance_score: 0.9953
- localization_score: 0.0199
- false_positive_score: 0.3754

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=25 fn=0 precision=0.7191 recall=1.0000
- OK3: gt=274 tp=273 fp=69 fn=1 precision=0.7982 recall=0.9964
- 加强筋打伤: gt=8 tp=2 fp=0 fn=6 precision=1.0000 recall=0.2500
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=4 fp=37 fn=14 precision=0.0976 recall=0.2222
- 浅划伤: gt=13 tp=0 fp=0 fn=13 precision=0.0000 recall=0.0000
- 漏背锡: gt=46 tp=30 fp=44 fn=14 precision=0.4054 recall=0.6522
- 碰伤: gt=269 tp=179 fp=120 fn=80 precision=0.5987 recall=0.6654
- 脏污: gt=42 tp=7 fp=2 fn=35 precision=0.7778 recall=0.1667
- 轮廓划伤: gt=84 tp=42 fp=32 fn=37 precision=0.5676 recall=0.5000
- 锡丝残留: gt=12 tp=0 fp=0 fn=12 precision=0.0000 recall=0.0000
- 锡尖: gt=27 tp=27 fp=33 fn=0 precision=0.4500 recall=1.0000
- 锡膏: gt=46 tp=31 fp=34 fn=14 precision=0.4769 recall=0.6739
