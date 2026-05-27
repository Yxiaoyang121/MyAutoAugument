# Validation Error Diagnosis

- Status: completed
- TP: 705
- FP: 348
- FN: 187
- Precision: 0.6695
- Recall: 0.7790

## Diagnosis Vector
- small_object_score: 0.2835
- low_contrast_score: 0.5227
- class_imbalance_score: 0.9953
- localization_score: 0.0144
- false_positive_score: 0.3305

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=22 fn=0 precision=0.7442 recall=1.0000
- OK3: gt=274 tp=267 fp=69 fn=7 precision=0.7946 recall=0.9745
- 加强筋打伤: gt=8 tp=6 fp=0 fn=2 precision=1.0000 recall=0.7500
- 开裂: gt=2 tp=0 fp=4 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=2 fp=11 fn=16 precision=0.1538 recall=0.1111
- 浅划伤: gt=13 tp=4 fp=0 fn=9 precision=1.0000 recall=0.3077
- 漏背锡: gt=46 tp=40 fp=48 fn=4 precision=0.4545 recall=0.8696
- 碰伤: gt=269 tp=199 fp=101 fn=65 precision=0.6633 recall=0.7398
- 脏污: gt=42 tp=19 fp=15 fn=23 precision=0.5588 recall=0.4524
- 轮廓划伤: gt=84 tp=47 fp=57 fn=31 precision=0.4519 recall=0.5595
- 锡丝残留: gt=12 tp=3 fp=1 fn=9 precision=0.7500 recall=0.2500
- 锡尖: gt=27 tp=23 fp=5 fn=4 precision=0.8214 recall=0.8519
- 锡膏: gt=46 tp=31 fp=15 fn=15 precision=0.6739 recall=0.6739
