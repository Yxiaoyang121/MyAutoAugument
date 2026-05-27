# Validation Error Diagnosis

- Status: completed
- TP: 641
- FP: 344
- FN: 255
- Precision: 0.6508
- Recall: 0.7083

## Diagnosis Vector
- small_object_score: 0.3701
- low_contrast_score: 0.5541
- class_imbalance_score: 0.9953
- localization_score: 0.0099
- false_positive_score: 0.3492

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=25 fn=0 precision=0.7191 recall=1.0000
- OK3: gt=274 tp=272 fp=67 fn=2 precision=0.8024 recall=0.9927
- 加强筋打伤: gt=8 tp=0 fp=0 fn=8 precision=0.0000 recall=0.0000
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=2 fp=35 fn=16 precision=0.0541 recall=0.1111
- 浅划伤: gt=13 tp=0 fp=0 fn=13 precision=0.0000 recall=0.0000
- 漏背锡: gt=46 tp=37 fp=75 fn=4 precision=0.3304 recall=0.8043
- 碰伤: gt=269 tp=147 fp=39 fn=121 precision=0.7903 recall=0.5465
- 脏污: gt=42 tp=21 fp=40 fn=21 precision=0.3443 recall=0.5000
- 轮廓划伤: gt=84 tp=48 fp=39 fn=34 precision=0.5517 recall=0.5714
- 锡丝残留: gt=12 tp=0 fp=0 fn=12 precision=0.0000 recall=0.0000
- 锡尖: gt=27 tp=24 fp=11 fn=3 precision=0.6857 recall=0.8889
- 锡膏: gt=46 tp=26 fp=13 fn=19 precision=0.6667 recall=0.5652
