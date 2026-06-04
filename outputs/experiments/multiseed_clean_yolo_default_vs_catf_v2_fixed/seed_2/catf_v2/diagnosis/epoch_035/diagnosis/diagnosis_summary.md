# Validation Error Diagnosis

- Status: completed
- TP: 699
- FP: 267
- FN: 191
- Precision: 0.7236
- Recall: 0.7724

## Diagnosis Vector
- small_object_score: 0.2818
- low_contrast_score: 0.5246
- class_imbalance_score: 0.9369
- localization_score: 0.0166
- false_positive_score: 0.2764

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=21 fn=0 precision=0.7529 recall=1.0000
- OK3: gt=274 tp=274 fp=69 fn=0 precision=0.7988 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=1 fn=1 precision=0.8750 recall=0.8750
- 开裂: gt=2 tp=2 fp=3 fn=0 precision=0.4000 recall=1.0000
- 油污: gt=18 tp=3 fp=11 fn=15 precision=0.2143 recall=0.1667
- 浅划伤: gt=13 tp=5 fp=0 fn=8 precision=1.0000 recall=0.3846
- 漏背锡: gt=46 tp=29 fp=20 fn=12 precision=0.5918 recall=0.6304
- 碰伤: gt=269 tp=189 fp=86 fn=76 precision=0.6873 recall=0.7026
- 脏污: gt=42 tp=18 fp=9 fn=22 precision=0.6667 recall=0.4286
- 轮廓划伤: gt=84 tp=47 fp=24 fn=33 precision=0.6620 recall=0.5595
- 锡丝残留: gt=12 tp=9 fp=5 fn=3 precision=0.6429 recall=0.7500
- 锡尖: gt=27 tp=22 fp=3 fn=5 precision=0.8800 recall=0.8148
- 锡膏: gt=46 tp=30 fp=15 fn=16 precision=0.6667 recall=0.6522
