# Validation Error Diagnosis

- Status: completed
- TP: 696
- FP: 275
- FN: 195
- Precision: 0.7168
- Recall: 0.7691

## Diagnosis Vector
- small_object_score: 0.3005
- low_contrast_score: 0.5377
- class_imbalance_score: 0.8786
- localization_score: 0.0155
- false_positive_score: 0.2832

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=22 fn=0 precision=0.7442 recall=1.0000
- OK3: gt=274 tp=274 fp=70 fn=0 precision=0.7965 recall=1.0000
- 加强筋打伤: gt=8 tp=4 fp=1 fn=4 precision=0.8000 recall=0.5000
- 开裂: gt=2 tp=1 fp=0 fn=1 precision=1.0000 recall=0.5000
- 油污: gt=18 tp=6 fp=18 fn=12 precision=0.2500 recall=0.3333
- 浅划伤: gt=13 tp=7 fp=3 fn=6 precision=0.7000 recall=0.5385
- 漏背锡: gt=46 tp=32 fp=17 fn=10 precision=0.6531 recall=0.6957
- 碰伤: gt=269 tp=177 fp=71 fn=84 precision=0.7137 recall=0.6580
- 脏污: gt=42 tp=21 fp=23 fn=21 precision=0.4773 recall=0.5000
- 轮廓划伤: gt=84 tp=50 fp=28 fn=33 precision=0.6410 recall=0.5952
- 锡丝残留: gt=12 tp=9 fp=8 fn=3 precision=0.5294 recall=0.7500
- 锡尖: gt=27 tp=23 fp=4 fn=4 precision=0.8519 recall=0.8519
- 锡膏: gt=46 tp=28 fp=10 fn=17 precision=0.7368 recall=0.6087
