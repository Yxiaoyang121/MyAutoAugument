# Validation Error Diagnosis

- Status: completed
- TP: 689
- FP: 337
- FN: 200
- Precision: 0.6715
- Recall: 0.7613

## Diagnosis Vector
- small_object_score: 0.2920
- low_contrast_score: 0.5095
- class_imbalance_score: 0.9953
- localization_score: 0.0177
- false_positive_score: 0.3285

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=26 fn=0 precision=0.7111 recall=1.0000
- OK3: gt=274 tp=272 fp=62 fn=2 precision=0.8144 recall=0.9927
- 加强筋打伤: gt=8 tp=6 fp=0 fn=2 precision=1.0000 recall=0.7500
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=2 fp=13 fn=16 precision=0.1333 recall=0.1111
- 浅划伤: gt=13 tp=0 fp=0 fn=13 precision=0.0000 recall=0.0000
- 漏背锡: gt=46 tp=34 fp=31 fn=9 precision=0.5231 recall=0.7391
- 碰伤: gt=269 tp=193 fp=122 fn=69 precision=0.6127 recall=0.7175
- 脏污: gt=42 tp=25 fp=47 fn=17 precision=0.3472 recall=0.5952
- 轮廓划伤: gt=84 tp=39 fp=22 fn=40 precision=0.6393 recall=0.4643
- 锡丝残留: gt=12 tp=0 fp=0 fn=12 precision=0.0000 recall=0.0000
- 锡尖: gt=27 tp=23 fp=5 fn=4 precision=0.8214 recall=0.8519
- 锡膏: gt=46 tp=31 fp=9 fn=14 precision=0.7750 recall=0.6739
