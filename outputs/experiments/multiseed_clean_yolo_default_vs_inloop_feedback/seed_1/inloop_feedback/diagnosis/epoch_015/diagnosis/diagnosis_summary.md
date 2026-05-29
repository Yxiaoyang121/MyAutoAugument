# Validation Error Diagnosis

- Status: completed
- TP: 677
- FP: 337
- FN: 213
- Precision: 0.6677
- Recall: 0.7481

## Diagnosis Vector
- small_object_score: 0.3124
- low_contrast_score: 0.5052
- class_imbalance_score: 0.9953
- localization_score: 0.0166
- false_positive_score: 0.3323

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=22 fn=0 precision=0.7442 recall=1.0000
- OK3: gt=274 tp=273 fp=71 fn=1 precision=0.7936 recall=0.9964
- 加强筋打伤: gt=8 tp=6 fp=4 fn=2 precision=0.6000 recall=0.7500
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=2 fp=18 fn=16 precision=0.1000 recall=0.1111
- 浅划伤: gt=13 tp=2 fp=0 fn=11 precision=1.0000 recall=0.1538
- 漏背锡: gt=46 tp=23 fp=13 fn=22 precision=0.6389 recall=0.5000
- 碰伤: gt=269 tp=198 fp=130 fn=67 precision=0.6037 recall=0.7361
- 脏污: gt=42 tp=21 fp=44 fn=18 precision=0.3231 recall=0.5000
- 轮廓划伤: gt=84 tp=28 fp=17 fn=49 precision=0.6222 recall=0.3333
- 锡丝残留: gt=12 tp=0 fp=0 fn=12 precision=0.0000 recall=0.0000
- 锡尖: gt=27 tp=23 fp=4 fn=4 precision=0.8519 recall=0.8519
- 锡膏: gt=46 tp=37 fp=14 fn=9 precision=0.7255 recall=0.8043
