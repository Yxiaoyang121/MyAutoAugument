# Validation Error Diagnosis

- Status: completed
- TP: 724
- FP: 323
- FN: 166
- Precision: 0.6915
- Recall: 0.8000

## Diagnosis Vector
- small_object_score: 0.2547
- low_contrast_score: 0.5627
- class_imbalance_score: 0.9953
- localization_score: 0.0166
- false_positive_score: 0.3085

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=23 fn=0 precision=0.7356 recall=1.0000
- OK3: gt=274 tp=274 fp=68 fn=0 precision=0.8012 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=2 fn=1 precision=0.7778 recall=0.8750
- 开裂: gt=2 tp=0 fp=3 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=5 fp=14 fn=12 precision=0.2632 recall=0.2778
- 浅划伤: gt=13 tp=6 fp=2 fn=7 precision=0.7500 recall=0.4615
- 漏背锡: gt=46 tp=38 fp=34 fn=3 precision=0.5278 recall=0.8261
- 碰伤: gt=269 tp=190 fp=81 fn=73 precision=0.7011 recall=0.7063
- 脏污: gt=42 tp=20 fp=19 fn=20 precision=0.5128 recall=0.4762
- 轮廓划伤: gt=84 tp=46 fp=34 fn=37 precision=0.5750 recall=0.5476
- 锡丝残留: gt=12 tp=11 fp=7 fn=1 precision=0.6111 recall=0.9167
- 锡尖: gt=27 tp=27 fp=9 fn=0 precision=0.7500 recall=1.0000
- 锡膏: gt=46 tp=36 fp=27 fn=10 precision=0.5714 recall=0.7826
