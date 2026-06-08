# Validation Error Diagnosis

- Status: completed
- TP: 663
- FP: 326
- FN: 217
- Precision: 0.6704
- Recall: 0.7326

## Diagnosis Vector
- small_object_score: 0.3260
- low_contrast_score: 0.4825
- class_imbalance_score: 0.9953
- localization_score: 0.0276
- false_positive_score: 0.3296

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=21 fn=0 precision=0.7529 recall=1.0000
- OK3: gt=274 tp=274 fp=67 fn=0 precision=0.8035 recall=1.0000
- 加强筋打伤: gt=8 tp=5 fp=2 fn=3 precision=0.7143 recall=0.6250
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=2 fp=26 fn=16 precision=0.0714 recall=0.1111
- 浅划伤: gt=13 tp=6 fp=1 fn=7 precision=0.8571 recall=0.4615
- 漏背锡: gt=46 tp=19 fp=14 fn=18 precision=0.5758 recall=0.4130
- 碰伤: gt=269 tp=182 fp=116 fn=81 precision=0.6107 recall=0.6766
- 脏污: gt=42 tp=14 fp=17 fn=27 precision=0.4516 recall=0.3333
- 轮廓划伤: gt=84 tp=39 fp=47 fn=37 precision=0.4535 recall=0.4643
- 锡丝残留: gt=12 tp=3 fp=3 fn=9 precision=0.5000 recall=0.2500
- 锡尖: gt=27 tp=23 fp=4 fn=4 precision=0.8519 recall=0.8519
- 锡膏: gt=46 tp=32 fp=8 fn=13 precision=0.8000 recall=0.6957
