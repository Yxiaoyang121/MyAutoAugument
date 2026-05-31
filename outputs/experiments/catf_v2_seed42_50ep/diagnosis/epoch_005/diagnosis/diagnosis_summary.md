# Validation Error Diagnosis

- Status: completed
- TP: 604
- FP: 684
- FN: 276
- Precision: 0.4689
- Recall: 0.6674

## Diagnosis Vector
- small_object_score: 0.4024
- low_contrast_score: 0.5274
- class_imbalance_score: 0.9953
- localization_score: 0.0276
- false_positive_score: 0.5311

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=35 fn=0 precision=0.6465 recall=1.0000
- OK3: gt=274 tp=274 fp=87 fn=0 precision=0.7590 recall=1.0000
- 加强筋打伤: gt=8 tp=0 fp=0 fn=8 precision=0.0000 recall=0.0000
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=3 fp=61 fn=14 precision=0.0469 recall=0.1667
- 浅划伤: gt=13 tp=0 fp=0 fn=13 precision=0.0000 recall=0.0000
- 漏背锡: gt=46 tp=30 fp=137 fn=7 precision=0.1796 recall=0.6522
- 碰伤: gt=269 tp=145 fp=113 fn=116 precision=0.5620 recall=0.5390
- 脏污: gt=42 tp=21 fp=206 fn=17 precision=0.0925 recall=0.5000
- 轮廓划伤: gt=84 tp=26 fp=12 fn=55 precision=0.6842 recall=0.3095
- 锡丝残留: gt=12 tp=0 fp=0 fn=12 precision=0.0000 recall=0.0000
- 锡尖: gt=27 tp=23 fp=22 fn=4 precision=0.5111 recall=0.8519
- 锡膏: gt=46 tp=18 fp=11 fn=28 precision=0.6207 recall=0.3913
