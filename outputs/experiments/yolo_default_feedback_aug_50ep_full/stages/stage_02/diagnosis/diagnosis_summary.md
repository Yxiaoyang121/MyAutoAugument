# Validation Error Diagnosis

- Status: completed
- TP: 683
- FP: 324
- FN: 200
- Precision: 0.6783
- Recall: 0.7547

## Diagnosis Vector
- small_object_score: 0.3107
- low_contrast_score: 0.5110
- class_imbalance_score: 0.9953
- localization_score: 0.0243
- false_positive_score: 0.3217

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=20 fn=0 precision=0.7619 recall=1.0000
- OK3: gt=274 tp=273 fp=62 fn=1 precision=0.8149 recall=0.9964
- 加强筋打伤: gt=8 tp=6 fp=0 fn=2 precision=1.0000 recall=0.7500
- 开裂: gt=2 tp=0 fp=8 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=5 fp=29 fn=13 precision=0.1471 recall=0.2778
- 浅划伤: gt=13 tp=4 fp=7 fn=7 precision=0.3636 recall=0.3077
- 漏背锡: gt=46 tp=28 fp=9 fn=13 precision=0.7568 recall=0.6087
- 碰伤: gt=269 tp=170 fp=83 fn=94 precision=0.6719 recall=0.6320
- 脏污: gt=42 tp=22 fp=38 fn=17 precision=0.3667 recall=0.5238
- 轮廓划伤: gt=84 tp=53 fp=49 fn=25 precision=0.5196 recall=0.6310
- 锡丝残留: gt=12 tp=0 fp=3 fn=12 precision=0.0000 recall=0.0000
- 锡尖: gt=27 tp=27 fp=4 fn=0 precision=0.8710 recall=1.0000
- 锡膏: gt=46 tp=31 fp=12 fn=14 precision=0.7209 recall=0.6739
