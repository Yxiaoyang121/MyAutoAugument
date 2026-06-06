# Validation Error Diagnosis

- Status: completed
- TP: 740
- FP: 295
- FN: 151
- Precision: 0.7150
- Recall: 0.8177

## Diagnosis Vector
- small_object_score: 0.2377
- low_contrast_score: 0.5225
- class_imbalance_score: 0.8591
- localization_score: 0.0155
- false_positive_score: 0.2850

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=19 fn=0 precision=0.7711 recall=1.0000
- OK3: gt=274 tp=274 fp=65 fn=0 precision=0.8083 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=3 fn=1 precision=0.7000 recall=0.8750
- 开裂: gt=2 tp=2 fp=1 fn=0 precision=0.6667 recall=1.0000
- 油污: gt=18 tp=7 fp=16 fn=11 precision=0.3043 recall=0.3889
- 浅划伤: gt=13 tp=10 fp=10 fn=3 precision=0.5000 recall=0.7692
- 漏背锡: gt=46 tp=35 fp=29 fn=6 precision=0.5469 recall=0.7609
- 碰伤: gt=269 tp=201 fp=76 fn=63 precision=0.7256 recall=0.7472
- 脏污: gt=42 tp=24 fp=12 fn=18 precision=0.6667 recall=0.5714
- 轮廓划伤: gt=84 tp=48 fp=39 fn=32 precision=0.5517 recall=0.5714
- 锡丝残留: gt=12 tp=12 fp=8 fn=0 precision=0.6000 recall=1.0000
- 锡尖: gt=27 tp=24 fp=3 fn=3 precision=0.8889 recall=0.8889
- 锡膏: gt=46 tp=32 fp=14 fn=14 precision=0.6957 recall=0.6957
