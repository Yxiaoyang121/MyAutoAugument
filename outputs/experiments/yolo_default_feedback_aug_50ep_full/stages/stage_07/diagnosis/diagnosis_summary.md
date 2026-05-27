# Validation Error Diagnosis

- Status: completed
- TP: 731
- FP: 314
- FN: 157
- Precision: 0.6995
- Recall: 0.8077

## Diagnosis Vector
- small_object_score: 0.2479
- low_contrast_score: 0.5369
- class_imbalance_score: 0.8786
- localization_score: 0.0188
- false_positive_score: 0.3005

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=21 fn=0 precision=0.7529 recall=1.0000
- OK3: gt=274 tp=274 fp=67 fn=0 precision=0.8035 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=1 fn=1 precision=0.8750 recall=0.8750
- 开裂: gt=2 tp=2 fp=10 fn=0 precision=0.1667 recall=1.0000
- 油污: gt=18 tp=6 fp=16 fn=12 precision=0.2727 recall=0.3333
- 浅划伤: gt=13 tp=7 fp=3 fn=6 precision=0.7000 recall=0.5385
- 漏背锡: gt=46 tp=34 fp=15 fn=10 precision=0.6939 recall=0.7391
- 碰伤: gt=269 tp=188 fp=81 fn=75 precision=0.6989 recall=0.6989
- 脏污: gt=42 tp=26 fp=33 fn=12 precision=0.4407 recall=0.6190
- 轮廓划伤: gt=84 tp=53 fp=36 fn=27 precision=0.5955 recall=0.6310
- 锡丝残留: gt=12 tp=10 fp=9 fn=2 precision=0.5263 recall=0.8333
- 锡尖: gt=27 tp=26 fp=6 fn=1 precision=0.8125 recall=0.9630
- 锡膏: gt=46 tp=34 fp=16 fn=11 precision=0.6800 recall=0.7391
