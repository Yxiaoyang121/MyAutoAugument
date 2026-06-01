# Validation Error Diagnosis

- Status: completed
- TP: 729
- FP: 272
- FN: 160
- Precision: 0.7283
- Recall: 0.8055

## Diagnosis Vector
- small_object_score: 0.2479
- low_contrast_score: 0.5687
- class_imbalance_score: 0.8591
- localization_score: 0.0177
- false_positive_score: 0.2717

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=20 fn=0 precision=0.7619 recall=1.0000
- OK3: gt=274 tp=272 fp=61 fn=2 precision=0.8168 recall=0.9927
- 加强筋打伤: gt=8 tp=7 fp=3 fn=1 precision=0.7000 recall=0.8750
- 开裂: gt=2 tp=2 fp=1 fn=0 precision=0.6667 recall=1.0000
- 油污: gt=18 tp=7 fp=7 fn=9 precision=0.5000 recall=0.3889
- 浅划伤: gt=13 tp=7 fp=7 fn=6 precision=0.5000 recall=0.5385
- 漏背锡: gt=46 tp=33 fp=28 fn=7 precision=0.5410 recall=0.7174
- 碰伤: gt=269 tp=192 fp=70 fn=72 precision=0.7328 recall=0.7138
- 脏污: gt=42 tp=28 fp=22 fn=11 precision=0.5600 recall=0.6667
- 轮廓划伤: gt=84 tp=52 fp=28 fn=32 precision=0.6500 recall=0.6190
- 锡丝残留: gt=12 tp=11 fp=6 fn=1 precision=0.6471 recall=0.9167
- 锡尖: gt=27 tp=23 fp=7 fn=4 precision=0.7667 recall=0.8519
- 锡膏: gt=46 tp=31 fp=12 fn=15 precision=0.7209 recall=0.6739
