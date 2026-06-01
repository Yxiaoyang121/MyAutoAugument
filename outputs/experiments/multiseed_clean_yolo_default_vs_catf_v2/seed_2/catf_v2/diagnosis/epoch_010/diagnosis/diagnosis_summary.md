# Validation Error Diagnosis

- Status: completed
- TP: 651
- FP: 355
- FN: 241
- Precision: 0.6471
- Recall: 0.7193

## Diagnosis Vector
- small_object_score: 0.3463
- low_contrast_score: 0.4803
- class_imbalance_score: 0.9953
- localization_score: 0.0144
- false_positive_score: 0.3529

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=19 fn=0 precision=0.7711 recall=1.0000
- OK3: gt=274 tp=272 fp=61 fn=2 precision=0.8168 recall=0.9927
- 加强筋打伤: gt=8 tp=1 fp=0 fn=7 precision=1.0000 recall=0.1250
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=2 fp=14 fn=16 precision=0.1250 recall=0.1111
- 浅划伤: gt=13 tp=0 fp=0 fn=13 precision=0.0000 recall=0.0000
- 漏背锡: gt=46 tp=27 fp=17 fn=16 precision=0.6136 recall=0.5870
- 碰伤: gt=269 tp=167 fp=95 fn=98 precision=0.6374 recall=0.6208
- 脏污: gt=42 tp=11 fp=17 fn=30 precision=0.3929 recall=0.2619
- 轮廓划伤: gt=84 tp=56 fp=119 fn=24 precision=0.3200 recall=0.6667
- 锡丝残留: gt=12 tp=0 fp=0 fn=12 precision=0.0000 recall=0.0000
- 锡尖: gt=27 tp=22 fp=5 fn=4 precision=0.8148 recall=0.8148
- 锡膏: gt=46 tp=29 fp=8 fn=17 precision=0.7838 recall=0.6304
