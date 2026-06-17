# Validation Error Diagnosis

- Status: completed
- TP: 732
- FP: 368
- FN: 162
- Precision: 0.6655
- Recall: 0.8088

## Diagnosis Vector
- small_object_score: 0.2411
- low_contrast_score: 0.5528
- class_imbalance_score: 0.9953
- localization_score: 0.0122
- false_positive_score: 0.3345

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=20 fn=0 precision=0.7619 recall=1.0000
- OK3: gt=274 tp=274 fp=70 fn=0 precision=0.7965 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=0 fn=1 precision=1.0000 recall=0.8750
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=9 fp=42 fn=9 precision=0.1765 recall=0.5000
- 浅划伤: gt=13 tp=4 fp=3 fn=9 precision=0.5714 recall=0.3077
- 漏背锡: gt=46 tp=38 fp=45 fn=4 precision=0.4578 recall=0.8261
- 碰伤: gt=269 tp=202 fp=98 fn=65 precision=0.6733 recall=0.7509
- 脏污: gt=42 tp=16 fp=15 fn=25 precision=0.5161 recall=0.3810
- 轮廓划伤: gt=84 tp=50 fp=31 fn=33 precision=0.6173 recall=0.5952
- 锡丝残留: gt=12 tp=8 fp=7 fn=4 precision=0.5333 recall=0.6667
- 锡尖: gt=27 tp=24 fp=9 fn=2 precision=0.7273 recall=0.8889
- 锡膏: gt=46 tp=36 fp=28 fn=8 precision=0.5625 recall=0.7826
