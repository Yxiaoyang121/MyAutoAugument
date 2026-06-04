# Validation Error Diagnosis

- Status: completed
- TP: 547
- FP: 251
- FN: 342
- Precision: 0.6855
- Recall: 0.6044

## Diagnosis Vector
- small_object_score: 0.4550
- low_contrast_score: 0.4999
- class_imbalance_score: 0.9953
- localization_score: 0.0177
- false_positive_score: 0.3145

## Issues
- small_object_low_recall severity=medium suggestions=tiling, object-aware-crop, scale, copy-paste, mild-geometry
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=16 fn=0 precision=0.8000 recall=1.0000
- OK3: gt=274 tp=266 fp=53 fn=8 precision=0.8339 recall=0.9708
- 加强筋打伤: gt=8 tp=0 fp=0 fn=8 precision=0.0000 recall=0.0000
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=0 fp=1 fn=18 precision=0.0000 recall=0.0000
- 浅划伤: gt=13 tp=4 fp=0 fn=9 precision=1.0000 recall=0.3077
- 漏背锡: gt=46 tp=5 fp=8 fn=38 precision=0.3846 recall=0.1087
- 碰伤: gt=269 tp=144 fp=91 fn=120 precision=0.6128 recall=0.5353
- 脏污: gt=42 tp=9 fp=49 fn=32 precision=0.1552 recall=0.2143
- 轮廓划伤: gt=84 tp=19 fp=9 fn=58 precision=0.6786 recall=0.2262
- 锡丝残留: gt=12 tp=0 fp=0 fn=12 precision=0.0000 recall=0.0000
- 锡尖: gt=27 tp=27 fp=19 fn=0 precision=0.5870 recall=1.0000
- 锡膏: gt=46 tp=9 fp=5 fn=37 precision=0.6429 recall=0.1957
