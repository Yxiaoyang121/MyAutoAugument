# Validation Error Diagnosis

- Status: completed
- TP: 693
- FP: 362
- FN: 197
- Precision: 0.6569
- Recall: 0.7657

## Diagnosis Vector
- small_object_score: 0.2818
- low_contrast_score: 0.4987
- class_imbalance_score: 0.9369
- localization_score: 0.0166
- false_positive_score: 0.3431

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=31 fn=0 precision=0.6737 recall=1.0000
- OK3: gt=274 tp=274 fp=70 fn=0 precision=0.7965 recall=1.0000
- 加强筋打伤: gt=8 tp=4 fp=0 fn=4 precision=1.0000 recall=0.5000
- 开裂: gt=2 tp=1 fp=1 fn=1 precision=0.5000 recall=0.5000
- 油污: gt=18 tp=3 fp=41 fn=15 precision=0.0682 recall=0.1667
- 浅划伤: gt=13 tp=7 fp=5 fn=5 precision=0.5833 recall=0.5385
- 漏背锡: gt=46 tp=33 fp=19 fn=12 precision=0.6346 recall=0.7174
- 碰伤: gt=269 tp=180 fp=94 fn=86 precision=0.6569 recall=0.6691
- 脏污: gt=42 tp=21 fp=19 fn=15 precision=0.5250 recall=0.5000
- 轮廓划伤: gt=84 tp=50 fp=66 fn=30 precision=0.4310 recall=0.5952
- 锡丝残留: gt=12 tp=6 fp=9 fn=6 precision=0.4000 recall=0.5000
- 锡尖: gt=27 tp=24 fp=4 fn=3 precision=0.8571 recall=0.8889
- 锡膏: gt=46 tp=26 fp=3 fn=20 precision=0.8966 recall=0.5652
