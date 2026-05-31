# Validation Error Diagnosis

- Status: completed
- TP: 698
- FP: 401
- FN: 193
- Precision: 0.6351
- Recall: 0.7713

## Diagnosis Vector
- small_object_score: 0.2716
- low_contrast_score: 0.5280
- class_imbalance_score: 0.9953
- localization_score: 0.0155
- false_positive_score: 0.3649

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=22 fn=0 precision=0.7442 recall=1.0000
- OK3: gt=274 tp=274 fp=65 fn=0 precision=0.8083 recall=1.0000
- 加强筋打伤: gt=8 tp=5 fp=0 fn=3 precision=1.0000 recall=0.6250
- 开裂: gt=2 tp=0 fp=5 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=4 fp=27 fn=14 precision=0.1290 recall=0.2222
- 浅划伤: gt=13 tp=4 fp=3 fn=9 precision=0.5714 recall=0.3077
- 漏背锡: gt=46 tp=27 fp=23 fn=14 precision=0.5400 recall=0.5870
- 碰伤: gt=269 tp=180 fp=98 fn=85 precision=0.6475 recall=0.6691
- 脏污: gt=42 tp=19 fp=55 fn=20 precision=0.2568 recall=0.4524
- 轮廓划伤: gt=84 tp=59 fp=39 fn=23 precision=0.6020 recall=0.7024
- 锡丝残留: gt=12 tp=9 fp=27 fn=3 precision=0.2500 recall=0.7500
- 锡尖: gt=27 tp=27 fp=15 fn=0 precision=0.6429 recall=1.0000
- 锡膏: gt=46 tp=26 fp=22 fn=20 precision=0.5417 recall=0.5652
