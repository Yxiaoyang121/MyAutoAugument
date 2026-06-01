# Validation Error Diagnosis

- Status: completed
- TP: 628
- FP: 362
- FN: 248
- Precision: 0.6343
- Recall: 0.6939

## Diagnosis Vector
- small_object_score: 0.3854
- low_contrast_score: 0.4976
- class_imbalance_score: 0.9953
- localization_score: 0.0320
- false_positive_score: 0.3657

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=26 fn=0 precision=0.7111 recall=1.0000
- OK3: gt=274 tp=272 fp=65 fn=2 precision=0.8071 recall=0.9927
- 加强筋打伤: gt=8 tp=0 fp=0 fn=8 precision=0.0000 recall=0.0000
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=3 fp=36 fn=11 precision=0.0769 recall=0.1667
- 浅划伤: gt=13 tp=0 fp=0 fn=13 precision=0.0000 recall=0.0000
- 漏背锡: gt=46 tp=30 fp=52 fn=12 precision=0.3659 recall=0.6522
- 碰伤: gt=269 tp=146 fp=81 fn=112 precision=0.6432 recall=0.5428
- 脏污: gt=42 tp=11 fp=23 fn=31 precision=0.3235 recall=0.2619
- 轮廓划伤: gt=84 tp=48 fp=50 fn=28 precision=0.4898 recall=0.5714
- 锡丝残留: gt=12 tp=0 fp=0 fn=12 precision=0.0000 recall=0.0000
- 锡尖: gt=27 tp=22 fp=9 fn=5 precision=0.7097 recall=0.8148
- 锡膏: gt=46 tp=32 fp=20 fn=12 precision=0.6154 recall=0.6957
