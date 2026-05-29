# Validation Error Diagnosis

- Status: completed
- TP: 704
- FP: 342
- FN: 185
- Precision: 0.6730
- Recall: 0.7779

## Diagnosis Vector
- small_object_score: 0.2869
- low_contrast_score: 0.5238
- class_imbalance_score: 0.8337
- localization_score: 0.0177
- false_positive_score: 0.3270

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=23 fn=0 precision=0.7356 recall=1.0000
- OK3: gt=274 tp=274 fp=71 fn=0 precision=0.7942 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=0 fn=1 precision=1.0000 recall=0.8750
- 开裂: gt=2 tp=2 fp=4 fn=0 precision=0.3333 recall=1.0000
- 油污: gt=18 tp=9 fp=53 fn=7 precision=0.1452 recall=0.5000
- 浅划伤: gt=13 tp=6 fp=0 fn=7 precision=1.0000 recall=0.4615
- 漏背锡: gt=46 tp=32 fp=41 fn=13 precision=0.4384 recall=0.6957
- 碰伤: gt=269 tp=178 fp=79 fn=87 precision=0.6926 recall=0.6617
- 脏污: gt=42 tp=22 fp=23 fn=16 precision=0.4889 recall=0.5238
- 轮廓划伤: gt=84 tp=44 fp=22 fn=35 precision=0.6667 recall=0.5238
- 锡丝残留: gt=12 tp=8 fp=8 fn=4 precision=0.5000 recall=0.6667
- 锡尖: gt=27 tp=27 fp=8 fn=0 precision=0.7714 recall=1.0000
- 锡膏: gt=46 tp=31 fp=10 fn=15 precision=0.7561 recall=0.6739
