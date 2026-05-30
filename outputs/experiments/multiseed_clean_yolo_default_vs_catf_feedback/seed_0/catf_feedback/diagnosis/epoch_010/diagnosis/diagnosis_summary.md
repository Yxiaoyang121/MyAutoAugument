# Validation Error Diagnosis

- Status: completed
- TP: 647
- FP: 346
- FN: 240
- Precision: 0.6516
- Recall: 0.7149

## Diagnosis Vector
- small_object_score: 0.3463
- low_contrast_score: 0.4852
- class_imbalance_score: 0.9953
- localization_score: 0.0199
- false_positive_score: 0.3484

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=29 fn=0 precision=0.6882 recall=1.0000
- OK3: gt=274 tp=272 fp=64 fn=2 precision=0.8095 recall=0.9927
- 加强筋打伤: gt=8 tp=1 fp=0 fn=7 precision=1.0000 recall=0.1250
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=2 fp=47 fn=16 precision=0.0408 recall=0.1111
- 浅划伤: gt=13 tp=6 fp=5 fn=7 precision=0.5455 recall=0.4615
- 漏背锡: gt=46 tp=19 fp=33 fn=19 precision=0.3654 recall=0.4130
- 碰伤: gt=269 tp=172 fp=72 fn=90 precision=0.7049 recall=0.6394
- 脏污: gt=42 tp=14 fp=21 fn=27 precision=0.4000 recall=0.3333
- 轮廓划伤: gt=84 tp=47 fp=24 fn=35 precision=0.6620 recall=0.5595
- 锡丝残留: gt=12 tp=0 fp=0 fn=12 precision=0.0000 recall=0.0000
- 锡尖: gt=27 tp=14 fp=0 fn=13 precision=1.0000 recall=0.5185
- 锡膏: gt=46 tp=36 fp=51 fn=10 precision=0.4138 recall=0.7826
