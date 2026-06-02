# Validation Error Diagnosis

- Status: completed
- TP: 705
- FP: 292
- FN: 184
- Precision: 0.7071
- Recall: 0.7790

## Diagnosis Vector
- small_object_score: 0.2767
- low_contrast_score: 0.5329
- class_imbalance_score: 0.9953
- localization_score: 0.0177
- false_positive_score: 0.2929

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=20 fn=0 precision=0.7619 recall=1.0000
- OK3: gt=274 tp=274 fp=67 fn=0 precision=0.8035 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=0 fn=1 precision=1.0000 recall=0.8750
- 开裂: gt=2 tp=0 fp=4 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=2 fp=13 fn=16 precision=0.1333 recall=0.1111
- 浅划伤: gt=13 tp=4 fp=2 fn=9 precision=0.6667 recall=0.3077
- 漏背锡: gt=46 tp=37 fp=31 fn=7 precision=0.5441 recall=0.8043
- 碰伤: gt=269 tp=190 fp=83 fn=75 precision=0.6960 recall=0.7063
- 脏污: gt=42 tp=22 fp=8 fn=18 precision=0.7333 recall=0.5238
- 轮廓划伤: gt=84 tp=45 fp=38 fn=31 precision=0.5422 recall=0.5357
- 锡丝残留: gt=12 tp=9 fp=4 fn=3 precision=0.6923 recall=0.7500
- 锡尖: gt=27 tp=19 fp=4 fn=8 precision=0.8261 recall=0.7037
- 锡膏: gt=46 tp=32 fp=18 fn=14 precision=0.6400 recall=0.6957
