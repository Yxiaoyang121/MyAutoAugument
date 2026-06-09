# Validation Error Diagnosis

- Status: completed
- TP: 286
- FP: 91
- FN: 153
- Precision: 0.7586
- Recall: 0.6471

## Diagnosis Vector
- small_object_score: 0.3913
- low_contrast_score: 0.4856
- class_imbalance_score: 0.9853
- localization_score: 0.0068
- false_positive_score: 0.2414

## Issues
- background_interference severity=medium suggestions=background-diversity, light-noise, illumination-jitter
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=38 fp=5 fn=0 precision=0.8837 recall=1.0000
- OK3: gt=102 tp=97 fp=19 fn=5 precision=0.8362 recall=0.9510
- 加强筋打伤: gt=6 tp=1 fp=0 fn=5 precision=1.0000 recall=0.1667
- 开裂: gt=3 tp=0 fp=0 fn=3 precision=0.0000 recall=0.0000
- 油污: gt=17 tp=1 fp=0 fn=16 precision=1.0000 recall=0.0588
- 浅划伤: gt=14 tp=1 fp=1 fn=13 precision=0.5000 recall=0.0714
- 漏背锡: gt=32 tp=24 fp=22 fn=8 precision=0.5217 recall=0.7500
- 碰伤: gt=133 tp=83 fp=24 fn=48 precision=0.7757 recall=0.6241
- 脏污: gt=28 tp=1 fp=0 fn=27 precision=1.0000 recall=0.0357
- 轮廓划伤: gt=30 tp=23 fp=6 fn=6 precision=0.7931 recall=0.7667
- 锡丝残留: gt=6 tp=0 fp=0 fn=6 precision=0.0000 recall=0.0000
- 锡尖: gt=12 tp=11 fp=14 fn=1 precision=0.4400 recall=0.9167
- 锡膏: gt=21 tp=6 fp=0 fn=15 precision=1.0000 recall=0.2857
