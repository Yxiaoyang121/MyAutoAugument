# Validation Error Diagnosis

- Status: completed
- TP: 609
- FP: 361
- FN: 279
- Precision: 0.6278
- Recall: 0.6729

## Diagnosis Vector
- small_object_score: 0.4058
- low_contrast_score: 0.4636
- class_imbalance_score: 0.9953
- localization_score: 0.0188
- false_positive_score: 0.3722

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=28 fn=0 precision=0.6957 recall=1.0000
- OK3: gt=274 tp=274 fp=67 fn=0 precision=0.8035 recall=1.0000
- 加强筋打伤: gt=8 tp=1 fp=0 fn=7 precision=1.0000 recall=0.1250
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=3 fp=68 fn=14 precision=0.0423 recall=0.1667
- 浅划伤: gt=13 tp=1 fp=14 fn=5 precision=0.0667 recall=0.0769
- 漏背锡: gt=46 tp=22 fp=14 fn=21 precision=0.6111 recall=0.4783
- 碰伤: gt=269 tp=169 fp=96 fn=94 precision=0.6377 recall=0.6283
- 脏污: gt=42 tp=0 fp=11 fn=42 precision=0.0000 recall=0.0000
- 轮廓划伤: gt=84 tp=32 fp=27 fn=52 precision=0.5424 recall=0.3810
- 锡丝残留: gt=12 tp=0 fp=0 fn=12 precision=0.0000 recall=0.0000
- 锡尖: gt=27 tp=8 fp=0 fn=19 precision=1.0000 recall=0.2963
- 锡膏: gt=46 tp=35 fp=36 fn=11 precision=0.4930 recall=0.7609
