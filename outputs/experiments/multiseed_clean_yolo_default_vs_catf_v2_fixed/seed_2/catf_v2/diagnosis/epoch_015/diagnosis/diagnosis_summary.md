# Validation Error Diagnosis

- Status: completed
- TP: 677
- FP: 291
- FN: 210
- Precision: 0.6994
- Recall: 0.7481

## Diagnosis Vector
- small_object_score: 0.2869
- low_contrast_score: 0.4636
- class_imbalance_score: 0.9953
- localization_score: 0.0199
- false_positive_score: 0.3006

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=22 fn=0 precision=0.7442 recall=1.0000
- OK3: gt=274 tp=272 fp=62 fn=2 precision=0.8144 recall=0.9927
- 加强筋打伤: gt=8 tp=2 fp=0 fn=6 precision=1.0000 recall=0.2500
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=2 fp=7 fn=16 precision=0.2222 recall=0.1111
- 浅划伤: gt=13 tp=1 fp=0 fn=12 precision=1.0000 recall=0.0769
- 漏背锡: gt=46 tp=18 fp=4 fn=27 precision=0.8182 recall=0.3913
- 碰伤: gt=269 tp=202 fp=124 fn=61 precision=0.6196 recall=0.7509
- 脏污: gt=42 tp=18 fp=17 fn=19 precision=0.5143 recall=0.4286
- 轮廓划伤: gt=84 tp=39 fp=25 fn=39 precision=0.6094 recall=0.4643
- 锡丝残留: gt=12 tp=0 fp=0 fn=12 precision=0.0000 recall=0.0000
- 锡尖: gt=27 tp=27 fp=7 fn=0 precision=0.7941 recall=1.0000
- 锡膏: gt=46 tp=32 fp=23 fn=14 precision=0.5818 recall=0.6957
