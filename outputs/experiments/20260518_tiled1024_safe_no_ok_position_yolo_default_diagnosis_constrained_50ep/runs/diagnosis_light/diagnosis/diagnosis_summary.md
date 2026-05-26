# Validation Error Diagnosis

- Status: completed
- TP: 720
- FP: 278
- FN: 172
- Precision: 0.7214
- Recall: 0.7956

## Diagnosis Vector
- small_object_score: 0.2649
- low_contrast_score: 0.5506
- class_imbalance_score: 0.9175
- localization_score: 0.0144
- false_positive_score: 0.2786

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=18 fn=0 precision=0.7805 recall=1.0000
- OK3: gt=274 tp=272 fp=66 fn=2 precision=0.8047 recall=0.9927
- 加强筋打伤: gt=8 tp=7 fp=0 fn=1 precision=1.0000 recall=0.8750
- 开裂: gt=2 tp=2 fp=3 fn=0 precision=0.4000 recall=1.0000
- 油污: gt=18 tp=4 fp=14 fn=14 precision=0.2222 recall=0.2222
- 浅划伤: gt=13 tp=5 fp=8 fn=8 precision=0.3846 recall=0.3846
- 漏背锡: gt=46 tp=34 fp=24 fn=5 precision=0.5862 recall=0.7391
- 碰伤: gt=269 tp=188 fp=75 fn=75 precision=0.7148 recall=0.6989
- 脏污: gt=42 tp=18 fp=7 fn=24 precision=0.7200 recall=0.4286
- 轮廓划伤: gt=84 tp=56 fp=45 fn=28 precision=0.5545 recall=0.6667
- 锡丝残留: gt=12 tp=11 fp=6 fn=1 precision=0.6471 recall=0.9167
- 锡尖: gt=27 tp=27 fp=7 fn=0 precision=0.7941 recall=1.0000
- 锡膏: gt=46 tp=32 fp=5 fn=14 precision=0.8649 recall=0.6957
