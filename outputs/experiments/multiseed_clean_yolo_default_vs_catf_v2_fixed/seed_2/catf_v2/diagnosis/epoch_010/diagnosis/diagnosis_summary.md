# Validation Error Diagnosis

- Status: completed
- TP: 595
- FP: 217
- FN: 296
- Precision: 0.7328
- Recall: 0.6575

## Diagnosis Vector
- small_object_score: 0.4278
- low_contrast_score: 0.4578
- class_imbalance_score: 0.9953
- localization_score: 0.0155
- false_positive_score: 0.2672

## Issues
- small_object_low_recall severity=medium suggestions=tiling, object-aware-crop, scale, copy-paste, mild-geometry
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=23 fn=0 precision=0.7356 recall=1.0000
- OK3: gt=274 tp=268 fp=65 fn=6 precision=0.8048 recall=0.9781
- 加强筋打伤: gt=8 tp=0 fp=0 fn=8 precision=0.0000 recall=0.0000
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=4 fp=13 fn=14 precision=0.2353 recall=0.2222
- 浅划伤: gt=13 tp=0 fp=0 fn=12 precision=0.0000 recall=0.0000
- 漏背锡: gt=46 tp=9 fp=1 fn=37 precision=0.9000 recall=0.1957
- 碰伤: gt=269 tp=160 fp=65 fn=106 precision=0.7111 recall=0.5948
- 脏污: gt=42 tp=12 fp=5 fn=26 precision=0.7059 recall=0.2857
- 轮廓划伤: gt=84 tp=41 fp=44 fn=37 precision=0.4824 recall=0.4881
- 锡丝残留: gt=12 tp=0 fp=0 fn=12 precision=0.0000 recall=0.0000
- 锡尖: gt=27 tp=13 fp=1 fn=14 precision=0.9286 recall=0.4815
- 锡膏: gt=46 tp=24 fp=0 fn=22 precision=1.0000 recall=0.5217
