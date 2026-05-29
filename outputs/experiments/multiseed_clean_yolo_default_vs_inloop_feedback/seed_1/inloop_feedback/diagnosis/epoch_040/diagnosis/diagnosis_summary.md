# Validation Error Diagnosis

- Status: completed
- TP: 727
- FP: 316
- FN: 159
- Precision: 0.6970
- Recall: 0.8033

## Diagnosis Vector
- small_object_score: 0.2564
- low_contrast_score: 0.5588
- class_imbalance_score: 0.9175
- localization_score: 0.0210
- false_positive_score: 0.3030

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=24 fn=0 precision=0.7273 recall=1.0000
- OK3: gt=274 tp=272 fp=67 fn=2 precision=0.8024 recall=0.9927
- 加强筋打伤: gt=8 tp=7 fp=0 fn=1 precision=1.0000 recall=0.8750
- 开裂: gt=2 tp=2 fp=5 fn=0 precision=0.2857 recall=1.0000
- 油污: gt=18 tp=4 fp=16 fn=14 precision=0.2000 recall=0.2222
- 浅划伤: gt=13 tp=6 fp=11 fn=5 precision=0.3529 recall=0.4615
- 漏背锡: gt=46 tp=35 fp=28 fn=7 precision=0.5556 recall=0.7609
- 碰伤: gt=269 tp=188 fp=84 fn=75 precision=0.6912 recall=0.6989
- 脏污: gt=42 tp=30 fp=21 fn=8 precision=0.5882 recall=0.7143
- 轮廓划伤: gt=84 tp=50 fp=33 fn=31 precision=0.6024 recall=0.5952
- 锡丝残留: gt=12 tp=12 fp=10 fn=0 precision=0.5455 recall=1.0000
- 锡尖: gt=27 tp=26 fp=4 fn=1 precision=0.8667 recall=0.9630
- 锡膏: gt=46 tp=31 fp=13 fn=15 precision=0.7045 recall=0.6739
