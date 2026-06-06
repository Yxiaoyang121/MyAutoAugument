# Validation Error Diagnosis

- Status: completed
- TP: 660
- FP: 332
- FN: 217
- Precision: 0.6653
- Recall: 0.7293

## Diagnosis Vector
- small_object_score: 0.3175
- low_contrast_score: 0.4717
- class_imbalance_score: 0.9953
- localization_score: 0.0309
- false_positive_score: 0.3347

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=25 fn=0 precision=0.7191 recall=1.0000
- OK3: gt=274 tp=269 fp=66 fn=5 precision=0.8030 recall=0.9818
- 加强筋打伤: gt=8 tp=2 fp=0 fn=6 precision=1.0000 recall=0.2500
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=0 fp=4 fn=16 precision=0.0000 recall=0.0000
- 浅划伤: gt=13 tp=0 fp=0 fn=13 precision=0.0000 recall=0.0000
- 漏背锡: gt=46 tp=33 fp=46 fn=6 precision=0.4177 recall=0.7174
- 碰伤: gt=269 tp=186 fp=113 fn=74 precision=0.6221 recall=0.6914
- 脏污: gt=42 tp=15 fp=15 fn=25 precision=0.5000 recall=0.3571
- 轮廓划伤: gt=84 tp=47 fp=51 fn=29 precision=0.4796 recall=0.5595
- 锡丝残留: gt=12 tp=0 fp=0 fn=12 precision=0.0000 recall=0.0000
- 锡尖: gt=27 tp=24 fp=4 fn=3 precision=0.8571 recall=0.8889
- 锡膏: gt=46 tp=20 fp=8 fn=26 precision=0.7143 recall=0.4348
