# Validation Error Diagnosis

- Status: completed
- TP: 710
- FP: 425
- FN: 171
- Precision: 0.6256
- Recall: 0.7845

## Diagnosis Vector
- small_object_score: 0.2547
- low_contrast_score: 0.4813
- class_imbalance_score: 0.9953
- localization_score: 0.0265
- false_positive_score: 0.3744

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=27 fn=0 precision=0.7033 recall=1.0000
- OK3: gt=274 tp=271 fp=67 fn=3 precision=0.8018 recall=0.9891
- 加强筋打伤: gt=8 tp=7 fp=0 fn=1 precision=1.0000 recall=0.8750
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=0 fp=9 fn=18 precision=0.0000 recall=0.0000
- 浅划伤: gt=13 tp=8 fp=22 fn=5 precision=0.2667 recall=0.6154
- 漏背锡: gt=46 tp=28 fp=56 fn=14 precision=0.3333 recall=0.6087
- 碰伤: gt=269 tp=202 fp=139 fn=59 precision=0.5924 recall=0.7509
- 脏污: gt=42 tp=20 fp=16 fn=19 precision=0.5556 recall=0.4762
- 轮廓划伤: gt=84 tp=51 fp=58 fn=24 precision=0.4679 recall=0.6071
- 锡丝残留: gt=12 tp=4 fp=3 fn=8 precision=0.5714 recall=0.3333
- 锡尖: gt=27 tp=27 fp=13 fn=0 precision=0.6750 recall=1.0000
- 锡膏: gt=46 tp=28 fp=15 fn=18 precision=0.6512 recall=0.6087
