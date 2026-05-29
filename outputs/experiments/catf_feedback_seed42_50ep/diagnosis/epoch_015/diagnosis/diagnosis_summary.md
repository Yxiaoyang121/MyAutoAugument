# Validation Error Diagnosis

- Status: completed
- TP: 667
- FP: 347
- FN: 217
- Precision: 0.6578
- Recall: 0.7370

## Diagnosis Vector
- small_object_score: 0.3175
- low_contrast_score: 0.4968
- class_imbalance_score: 0.9953
- localization_score: 0.0232
- false_positive_score: 0.3422

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=61 fp=14 fn=3 precision=0.8133 recall=0.9531
- OK3: gt=274 tp=274 fp=71 fn=0 precision=0.7942 recall=1.0000
- 加强筋打伤: gt=8 tp=6 fp=2 fn=2 precision=0.7500 recall=0.7500
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=6 fp=21 fn=12 precision=0.2222 recall=0.3333
- 浅划伤: gt=13 tp=5 fp=0 fn=8 precision=1.0000 recall=0.3846
- 漏背锡: gt=46 tp=26 fp=21 fn=16 precision=0.5532 recall=0.5652
- 碰伤: gt=269 tp=168 fp=77 fn=95 precision=0.6857 recall=0.6245
- 脏污: gt=42 tp=15 fp=44 fn=21 precision=0.2542 recall=0.3571
- 轮廓划伤: gt=84 tp=50 fp=49 fn=31 precision=0.5051 recall=0.5952
- 锡丝残留: gt=12 tp=0 fp=0 fn=12 precision=0.0000 recall=0.0000
- 锡尖: gt=27 tp=25 fp=13 fn=2 precision=0.6579 recall=0.9259
- 锡膏: gt=46 tp=31 fp=35 fn=13 precision=0.4697 recall=0.6739
