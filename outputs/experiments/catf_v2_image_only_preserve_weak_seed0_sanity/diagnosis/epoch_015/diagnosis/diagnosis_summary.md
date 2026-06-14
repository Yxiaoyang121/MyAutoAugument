# Validation Error Diagnosis

- Status: completed
- TP: 682
- FP: 439
- FN: 206
- Precision: 0.6084
- Recall: 0.7536

## Diagnosis Vector
- small_object_score: 0.3158
- low_contrast_score: 0.5434
- class_imbalance_score: 0.9953
- localization_score: 0.0188
- false_positive_score: 0.3916

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=27 fn=0 precision=0.7033 recall=1.0000
- OK3: gt=274 tp=273 fp=68 fn=1 precision=0.8006 recall=0.9964
- 加强筋打伤: gt=8 tp=7 fp=0 fn=1 precision=1.0000 recall=0.8750
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=2 fp=17 fn=15 precision=0.1053 recall=0.1111
- 浅划伤: gt=13 tp=3 fp=0 fn=10 precision=1.0000 recall=0.2308
- 漏背锡: gt=46 tp=26 fp=33 fn=17 precision=0.4407 recall=0.5652
- 碰伤: gt=269 tp=170 fp=107 fn=95 precision=0.6137 recall=0.6320
- 脏污: gt=42 tp=17 fp=134 fn=20 precision=0.1126 recall=0.4048
- 轮廓划伤: gt=84 tp=53 fp=32 fn=27 precision=0.6235 recall=0.6310
- 锡丝残留: gt=12 tp=6 fp=2 fn=6 precision=0.7500 recall=0.5000
- 锡尖: gt=27 tp=27 fp=13 fn=0 precision=0.6750 recall=1.0000
- 锡膏: gt=46 tp=34 fp=6 fn=12 precision=0.8500 recall=0.7391
