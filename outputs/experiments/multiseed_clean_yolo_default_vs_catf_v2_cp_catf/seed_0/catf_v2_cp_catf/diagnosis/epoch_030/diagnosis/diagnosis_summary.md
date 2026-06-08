# Validation Error Diagnosis

- Status: completed
- TP: 729
- FP: 394
- FN: 160
- Precision: 0.6492
- Recall: 0.8055

## Diagnosis Vector
- small_object_score: 0.2428
- low_contrast_score: 0.5363
- class_imbalance_score: 0.9953
- localization_score: 0.0177
- false_positive_score: 0.3508

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=23 fn=0 precision=0.7356 recall=1.0000
- OK3: gt=274 tp=274 fp=67 fn=0 precision=0.8035 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=6 fn=1 precision=0.5385 recall=0.8750
- 开裂: gt=2 tp=0 fp=3 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=8 fp=66 fn=7 precision=0.1081 recall=0.4444
- 浅划伤: gt=13 tp=6 fp=0 fn=7 precision=1.0000 recall=0.4615
- 漏背锡: gt=46 tp=36 fp=29 fn=6 precision=0.5538 recall=0.7826
- 碰伤: gt=269 tp=190 fp=85 fn=74 precision=0.6909 recall=0.7063
- 脏污: gt=42 tp=27 fp=55 fn=13 precision=0.3293 recall=0.6429
- 轮廓划伤: gt=84 tp=51 fp=28 fn=31 precision=0.6456 recall=0.6071
- 锡丝残留: gt=12 tp=9 fp=6 fn=3 precision=0.6000 recall=0.7500
- 锡尖: gt=27 tp=27 fp=8 fn=0 precision=0.7714 recall=1.0000
- 锡膏: gt=46 tp=30 fp=18 fn=16 precision=0.6250 recall=0.6522
