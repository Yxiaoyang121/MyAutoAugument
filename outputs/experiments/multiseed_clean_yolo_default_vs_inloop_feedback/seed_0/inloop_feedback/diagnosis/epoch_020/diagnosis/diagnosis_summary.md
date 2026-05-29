# Validation Error Diagnosis

- Status: completed
- TP: 717
- FP: 433
- FN: 164
- Precision: 0.6235
- Recall: 0.7923

## Diagnosis Vector
- small_object_score: 0.2547
- low_contrast_score: 0.5585
- class_imbalance_score: 0.9953
- localization_score: 0.0265
- false_positive_score: 0.3765

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=25 fn=0 precision=0.7191 recall=1.0000
- OK3: gt=274 tp=274 fp=69 fn=0 precision=0.7988 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=0 fn=1 precision=1.0000 recall=0.8750
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=6 fp=38 fn=12 precision=0.1364 recall=0.3333
- 浅划伤: gt=13 tp=4 fp=2 fn=6 precision=0.6667 recall=0.3077
- 漏背锡: gt=46 tp=28 fp=32 fn=7 precision=0.4667 recall=0.6087
- 碰伤: gt=269 tp=191 fp=99 fn=73 precision=0.6586 recall=0.7100
- 脏污: gt=42 tp=22 fp=21 fn=20 precision=0.5116 recall=0.5238
- 轮廓划伤: gt=84 tp=52 fp=47 fn=31 precision=0.5253 recall=0.6190
- 锡丝残留: gt=12 tp=7 fp=6 fn=5 precision=0.5385 recall=0.5833
- 锡尖: gt=27 tp=26 fp=14 fn=1 precision=0.6500 recall=0.9630
- 锡膏: gt=46 tp=36 fp=80 fn=6 precision=0.3103 recall=0.7826
