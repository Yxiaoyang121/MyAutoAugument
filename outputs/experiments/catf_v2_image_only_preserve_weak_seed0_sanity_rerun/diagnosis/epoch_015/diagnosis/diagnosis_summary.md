# Validation Error Diagnosis

- Status: completed
- TP: 672
- FP: 399
- FN: 208
- Precision: 0.6275
- Recall: 0.7425

## Diagnosis Vector
- small_object_score: 0.3277
- low_contrast_score: 0.4938
- class_imbalance_score: 0.9953
- localization_score: 0.0276
- false_positive_score: 0.3725

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=20 fn=0 precision=0.7619 recall=1.0000
- OK3: gt=274 tp=274 fp=69 fn=0 precision=0.7988 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=1 fn=1 precision=0.8750 recall=0.8750
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=5 fp=38 fn=13 precision=0.1163 recall=0.2778
- 浅划伤: gt=13 tp=4 fp=0 fn=9 precision=1.0000 recall=0.3077
- 漏背锡: gt=46 tp=28 fp=31 fn=10 precision=0.4746 recall=0.6087
- 碰伤: gt=269 tp=176 fp=103 fn=89 precision=0.6308 recall=0.6543
- 脏污: gt=42 tp=12 fp=54 fn=27 precision=0.1818 recall=0.2857
- 轮廓划伤: gt=84 tp=51 fp=62 fn=27 precision=0.4513 recall=0.6071
- 锡丝残留: gt=12 tp=3 fp=6 fn=9 precision=0.3333 recall=0.2500
- 锡尖: gt=27 tp=21 fp=8 fn=3 precision=0.7241 recall=0.7778
- 锡膏: gt=46 tp=27 fp=7 fn=18 precision=0.7941 recall=0.5870
