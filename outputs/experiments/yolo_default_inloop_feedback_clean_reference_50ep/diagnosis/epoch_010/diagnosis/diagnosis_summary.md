# Validation Error Diagnosis

- Status: completed
- TP: 575
- FP: 198
- FN: 316
- Precision: 0.7439
- Recall: 0.6354

## Diagnosis Vector
- small_object_score: 0.4380
- low_contrast_score: 0.4533
- class_imbalance_score: 0.9953
- localization_score: 0.0155
- false_positive_score: 0.2561

## Issues
- small_object_low_recall severity=medium suggestions=tiling, object-aware-crop, scale, copy-paste, mild-geometry
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=14 fn=0 precision=0.8205 recall=1.0000
- OK3: gt=274 tp=268 fp=60 fn=6 precision=0.8171 recall=0.9781
- 加强筋打伤: gt=8 tp=0 fp=0 fn=8 precision=0.0000 recall=0.0000
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=4 fp=14 fn=14 precision=0.2222 recall=0.2222
- 浅划伤: gt=13 tp=3 fp=0 fn=10 precision=1.0000 recall=0.2308
- 漏背锡: gt=46 tp=15 fp=13 fn=29 precision=0.5357 recall=0.3261
- 碰伤: gt=269 tp=149 fp=55 fn=110 precision=0.7304 recall=0.5539
- 脏污: gt=42 tp=7 fp=14 fn=35 precision=0.3333 recall=0.1667
- 轮廓划伤: gt=84 tp=37 fp=26 fn=45 precision=0.5873 recall=0.4405
- 锡丝残留: gt=12 tp=0 fp=0 fn=12 precision=0.0000 recall=0.0000
- 锡尖: gt=27 tp=18 fp=0 fn=9 precision=1.0000 recall=0.6667
- 锡膏: gt=46 tp=10 fp=2 fn=36 precision=0.8333 recall=0.2174
