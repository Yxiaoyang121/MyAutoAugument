# Validation Error Diagnosis

- Status: completed
- TP: 716
- FP: 386
- FN: 165
- Precision: 0.6497
- Recall: 0.7912

## Diagnosis Vector
- small_object_score: 0.2750
- low_contrast_score: 0.5512
- class_imbalance_score: 0.9953
- localization_score: 0.0265
- false_positive_score: 0.3503

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=24 fn=0 precision=0.7273 recall=1.0000
- OK3: gt=274 tp=274 fp=76 fn=0 precision=0.7829 recall=1.0000
- 加强筋打伤: gt=8 tp=5 fp=1 fn=3 precision=0.8333 recall=0.6250
- 开裂: gt=2 tp=0 fp=2 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=8 fp=37 fn=10 precision=0.1778 recall=0.4444
- 浅划伤: gt=13 tp=6 fp=0 fn=7 precision=1.0000 recall=0.4615
- 漏背锡: gt=46 tp=35 fp=57 fn=6 precision=0.3804 recall=0.7609
- 碰伤: gt=269 tp=180 fp=92 fn=82 precision=0.6618 recall=0.6691
- 脏污: gt=42 tp=21 fp=15 fn=18 precision=0.5833 recall=0.5000
- 轮廓划伤: gt=84 tp=51 fp=35 fn=26 precision=0.5930 recall=0.6071
- 锡丝残留: gt=12 tp=9 fp=4 fn=3 precision=0.6923 recall=0.7500
- 锡尖: gt=27 tp=24 fp=6 fn=2 precision=0.8000 recall=0.8889
- 锡膏: gt=46 tp=39 fp=37 fn=6 precision=0.5132 recall=0.8478
