# Validation Error Diagnosis

- Status: completed
- TP: 702
- FP: 379
- FN: 183
- Precision: 0.6494
- Recall: 0.7757

## Diagnosis Vector
- small_object_score: 0.2886
- low_contrast_score: 0.5391
- class_imbalance_score: 0.8786
- localization_score: 0.0221
- false_positive_score: 0.3506

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=28 fn=0 precision=0.6957 recall=1.0000
- OK3: gt=274 tp=274 fp=70 fn=0 precision=0.7965 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=1 fn=1 precision=0.8750 recall=0.8750
- 开裂: gt=2 tp=1 fp=13 fn=1 precision=0.0714 recall=0.5000
- 油污: gt=18 tp=6 fp=55 fn=9 precision=0.0984 recall=0.3333
- 浅划伤: gt=13 tp=6 fp=2 fn=7 precision=0.7500 recall=0.4615
- 漏背锡: gt=46 tp=31 fp=38 fn=11 precision=0.4493 recall=0.6739
- 碰伤: gt=269 tp=183 fp=77 fn=81 precision=0.7038 recall=0.6803
- 脏污: gt=42 tp=22 fp=27 fn=19 precision=0.4490 recall=0.5238
- 轮廓划伤: gt=84 tp=42 fp=40 fn=36 precision=0.5122 recall=0.5000
- 锡丝残留: gt=12 tp=7 fp=7 fn=5 precision=0.5000 recall=0.5833
- 锡尖: gt=27 tp=25 fp=5 fn=2 precision=0.8333 recall=0.9259
- 锡膏: gt=46 tp=34 fp=16 fn=11 precision=0.6800 recall=0.7391
