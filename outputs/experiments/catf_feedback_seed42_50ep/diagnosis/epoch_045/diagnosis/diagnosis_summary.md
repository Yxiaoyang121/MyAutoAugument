# Validation Error Diagnosis

- Status: completed
- TP: 709
- FP: 270
- FN: 180
- Precision: 0.7242
- Recall: 0.7834

## Diagnosis Vector
- small_object_score: 0.2852
- low_contrast_score: 0.5647
- class_imbalance_score: 0.8786
- localization_score: 0.0177
- false_positive_score: 0.2758

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=19 fn=0 precision=0.7711 recall=1.0000
- OK3: gt=274 tp=272 fp=66 fn=2 precision=0.8047 recall=0.9927
- 加强筋打伤: gt=8 tp=7 fp=0 fn=1 precision=1.0000 recall=0.8750
- 开裂: gt=2 tp=2 fp=3 fn=0 precision=0.4000 recall=1.0000
- 油污: gt=18 tp=6 fp=15 fn=12 precision=0.2857 recall=0.3333
- 浅划伤: gt=13 tp=8 fp=3 fn=5 precision=0.7273 recall=0.6154
- 漏背锡: gt=46 tp=30 fp=13 fn=12 precision=0.6977 recall=0.6522
- 碰伤: gt=269 tp=173 fp=74 fn=91 precision=0.7004 recall=0.6431
- 脏污: gt=42 tp=26 fp=20 fn=13 precision=0.5652 recall=0.6190
- 轮廓划伤: gt=84 tp=52 fp=35 fn=28 precision=0.5977 recall=0.6190
- 锡丝残留: gt=12 tp=10 fp=7 fn=2 precision=0.5882 recall=0.8333
- 锡尖: gt=27 tp=27 fp=3 fn=0 precision=0.9000 recall=1.0000
- 锡膏: gt=46 tp=32 fp=12 fn=14 precision=0.7273 recall=0.6957
