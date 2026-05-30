# Validation Error Diagnosis

- Status: completed
- TP: 685
- FP: 361
- FN: 193
- Precision: 0.6549
- Recall: 0.7569

## Diagnosis Vector
- small_object_score: 0.3158
- low_contrast_score: 0.5813
- class_imbalance_score: 0.9953
- localization_score: 0.0298
- false_positive_score: 0.3451

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=24 fn=0 precision=0.7273 recall=1.0000
- OK3: gt=274 tp=274 fp=71 fn=0 precision=0.7942 recall=1.0000
- 加强筋打伤: gt=8 tp=5 fp=0 fn=3 precision=1.0000 recall=0.6250
- 开裂: gt=2 tp=0 fp=2 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=4 fp=42 fn=14 precision=0.0870 recall=0.2222
- 浅划伤: gt=13 tp=5 fp=2 fn=8 precision=0.7143 recall=0.3846
- 漏背锡: gt=46 tp=30 fp=60 fn=4 precision=0.3333 recall=0.6522
- 碰伤: gt=269 tp=169 fp=88 fn=92 precision=0.6576 recall=0.6283
- 脏污: gt=42 tp=21 fp=7 fn=20 precision=0.7500 recall=0.5000
- 轮廓划伤: gt=84 tp=44 fp=20 fn=37 precision=0.6875 recall=0.5238
- 锡丝残留: gt=12 tp=6 fp=6 fn=5 precision=0.5000 recall=0.5000
- 锡尖: gt=27 tp=26 fp=12 fn=1 precision=0.6842 recall=0.9630
- 锡膏: gt=46 tp=37 fp=27 fn=7 precision=0.5781 recall=0.8043
