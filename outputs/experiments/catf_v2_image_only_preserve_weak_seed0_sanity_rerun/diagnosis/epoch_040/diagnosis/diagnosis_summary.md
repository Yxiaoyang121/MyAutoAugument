# Validation Error Diagnosis

- Status: completed
- TP: 729
- FP: 315
- FN: 163
- Precision: 0.6983
- Recall: 0.8055

## Diagnosis Vector
- small_object_score: 0.2496
- low_contrast_score: 0.5374
- class_imbalance_score: 0.8640
- localization_score: 0.0144
- false_positive_score: 0.3017

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=22 fn=0 precision=0.7442 recall=1.0000
- OK3: gt=274 tp=272 fp=69 fn=2 precision=0.7977 recall=0.9927
- 加强筋打伤: gt=8 tp=3 fp=0 fn=5 precision=1.0000 recall=0.3750
- 开裂: gt=2 tp=2 fp=5 fn=0 precision=0.2857 recall=1.0000
- 油污: gt=18 tp=7 fp=7 fn=11 precision=0.5000 recall=0.3889
- 浅划伤: gt=13 tp=8 fp=6 fn=4 precision=0.5714 recall=0.6154
- 漏背锡: gt=46 tp=31 fp=31 fn=11 precision=0.5000 recall=0.6739
- 碰伤: gt=269 tp=195 fp=85 fn=68 precision=0.6964 recall=0.7249
- 脏污: gt=42 tp=25 fp=27 fn=16 precision=0.4808 recall=0.5952
- 轮廓划伤: gt=84 tp=56 fp=31 fn=27 precision=0.6437 recall=0.6667
- 锡丝残留: gt=12 tp=9 fp=6 fn=3 precision=0.6000 recall=0.7500
- 锡尖: gt=27 tp=27 fp=11 fn=0 precision=0.7105 recall=1.0000
- 锡膏: gt=46 tp=30 fp=15 fn=16 precision=0.6667 recall=0.6522
