# Validation Error Diagnosis

- Status: completed
- TP: 626
- FP: 257
- FN: 272
- Precision: 0.7089
- Recall: 0.6917

## Diagnosis Vector
- small_object_score: 0.3990
- low_contrast_score: 0.5167
- class_imbalance_score: 0.9953
- localization_score: 0.0077
- false_positive_score: 0.2911

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=21 fn=0 precision=0.7529 recall=1.0000
- OK3: gt=274 tp=272 fp=66 fn=2 precision=0.8047 recall=0.9927
- 加强筋打伤: gt=8 tp=6 fp=1 fn=2 precision=0.8571 recall=0.7500
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=2 fp=5 fn=16 precision=0.2857 recall=0.1111
- 浅划伤: gt=13 tp=2 fp=0 fn=11 precision=1.0000 recall=0.1538
- 漏背锡: gt=46 tp=20 fp=16 fn=25 precision=0.5556 recall=0.4348
- 碰伤: gt=269 tp=147 fp=49 fn=119 precision=0.7500 recall=0.5465
- 脏污: gt=42 tp=21 fp=68 fn=19 precision=0.2360 recall=0.5000
- 轮廓划伤: gt=84 tp=36 fp=14 fn=47 precision=0.7200 recall=0.4286
- 锡丝残留: gt=12 tp=0 fp=0 fn=12 precision=0.0000 recall=0.0000
- 锡尖: gt=27 tp=21 fp=2 fn=6 precision=0.9130 recall=0.7778
- 锡膏: gt=46 tp=35 fp=15 fn=11 precision=0.7000 recall=0.7609
