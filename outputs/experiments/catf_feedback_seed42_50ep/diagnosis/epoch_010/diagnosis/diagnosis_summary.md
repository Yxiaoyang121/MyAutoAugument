# Validation Error Diagnosis

- Status: completed
- TP: 594
- FP: 218
- FN: 303
- Precision: 0.7315
- Recall: 0.6564

## Diagnosis Vector
- small_object_score: 0.4143
- low_contrast_score: 0.4848
- class_imbalance_score: 0.9953
- localization_score: 0.0088
- false_positive_score: 0.2685

## Issues
- small_object_low_recall severity=medium suggestions=tiling, object-aware-crop, scale, copy-paste, mild-geometry
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=21 fn=0 precision=0.7529 recall=1.0000
- OK3: gt=274 tp=273 fp=54 fn=1 precision=0.8349 recall=0.9964
- 加强筋打伤: gt=8 tp=0 fp=0 fn=8 precision=0.0000 recall=0.0000
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=3 fp=9 fn=15 precision=0.2500 recall=0.1667
- 浅划伤: gt=13 tp=3 fp=2 fn=10 precision=0.6000 recall=0.2308
- 漏背锡: gt=46 tp=16 fp=9 fn=29 precision=0.6400 recall=0.3478
- 碰伤: gt=269 tp=145 fp=78 fn=119 precision=0.6502 recall=0.5390
- 脏污: gt=42 tp=6 fp=2 fn=36 precision=0.7500 recall=0.1429
- 轮廓划伤: gt=84 tp=37 fp=35 fn=45 precision=0.5139 recall=0.4405
- 锡丝残留: gt=12 tp=0 fp=0 fn=12 precision=0.0000 recall=0.0000
- 锡尖: gt=27 tp=23 fp=1 fn=4 precision=0.9583 recall=0.8519
- 锡膏: gt=46 tp=24 fp=7 fn=22 precision=0.7742 recall=0.5217
