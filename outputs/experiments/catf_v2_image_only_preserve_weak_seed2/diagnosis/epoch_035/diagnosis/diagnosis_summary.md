# Validation Error Diagnosis

- Status: completed
- TP: 697
- FP: 256
- FN: 186
- Precision: 0.7314
- Recall: 0.7702

## Diagnosis Vector
- small_object_score: 0.2835
- low_contrast_score: 0.5231
- class_imbalance_score: 0.9564
- localization_score: 0.0243
- false_positive_score: 0.2686

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=21 fn=0 precision=0.7529 recall=1.0000
- OK3: gt=274 tp=274 fp=66 fn=0 precision=0.8059 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=2 fn=1 precision=0.7778 recall=0.8750
- 开裂: gt=2 tp=2 fp=2 fn=0 precision=0.5000 recall=1.0000
- 油污: gt=18 tp=2 fp=13 fn=16 precision=0.1333 recall=0.1111
- 浅划伤: gt=13 tp=4 fp=2 fn=9 precision=0.6667 recall=0.3077
- 漏背锡: gt=46 tp=30 fp=12 fn=11 precision=0.7143 recall=0.6522
- 碰伤: gt=269 tp=192 fp=76 fn=73 precision=0.7164 recall=0.7138
- 脏污: gt=42 tp=19 fp=15 fn=19 precision=0.5588 recall=0.4524
- 轮廓划伤: gt=84 tp=42 fp=25 fn=33 precision=0.6269 recall=0.5000
- 锡丝残留: gt=12 tp=9 fp=6 fn=3 precision=0.6000 recall=0.7500
- 锡尖: gt=27 tp=21 fp=2 fn=6 precision=0.9130 recall=0.7778
- 锡膏: gt=46 tp=31 fp=14 fn=15 precision=0.6889 recall=0.6739
