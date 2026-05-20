# Validation Error Diagnosis

- Status: completed
- TP: 636
- FP: 128
- FN: 257
- Precision: 0.8325
- Recall: 0.7028

## Diagnosis Vector
- small_object_score: 0.3803
- low_contrast_score: 0.5802
- class_imbalance_score: 0.9953
- localization_score: 0.0133
- false_positive_score: 0.1675

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=2 fn=0 precision=0.9697 recall=1.0000
- OK3: gt=274 tp=268 fp=8 fn=6 precision=0.9710 recall=0.9781
- 加强筋打伤: gt=8 tp=7 fp=0 fn=1 precision=1.0000 recall=0.8750
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=5 fp=29 fn=13 precision=0.1471 recall=0.2778
- 浅划伤: gt=13 tp=7 fp=5 fn=6 precision=0.5833 recall=0.5385
- 漏背锡: gt=46 tp=28 fp=13 fn=13 precision=0.6829 recall=0.6087
- 碰伤: gt=269 tp=127 fp=26 fn=136 precision=0.8301 recall=0.4721
- 脏污: gt=42 tp=20 fp=6 fn=22 precision=0.7692 recall=0.4762
- 轮廓划伤: gt=84 tp=47 fp=17 fn=36 precision=0.7344 recall=0.5595
- 锡丝残留: gt=12 tp=9 fp=3 fn=3 precision=0.7500 recall=0.7500
- 锡尖: gt=27 tp=25 fp=9 fn=2 precision=0.7353 recall=0.9259
- 锡膏: gt=46 tp=29 fp=10 fn=17 precision=0.7436 recall=0.6304
