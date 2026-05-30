# Validation Error Diagnosis

- Status: completed
- TP: 724
- FP: 301
- FN: 167
- Precision: 0.7063
- Recall: 0.8000

## Diagnosis Vector
- small_object_score: 0.2683
- low_contrast_score: 0.5470
- class_imbalance_score: 0.9953
- localization_score: 0.0155
- false_positive_score: 0.2937

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=19 fn=0 precision=0.7711 recall=1.0000
- OK3: gt=274 tp=272 fp=66 fn=2 precision=0.8047 recall=0.9927
- 加强筋打伤: gt=8 tp=7 fp=3 fn=1 precision=0.7000 recall=0.8750
- 开裂: gt=2 tp=0 fp=6 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=5 fp=22 fn=13 precision=0.1852 recall=0.2778
- 浅划伤: gt=13 tp=8 fp=8 fn=5 precision=0.5000 recall=0.6154
- 漏背锡: gt=46 tp=37 fp=26 fn=6 precision=0.5873 recall=0.8043
- 碰伤: gt=269 tp=186 fp=90 fn=76 precision=0.6739 recall=0.6914
- 脏污: gt=42 tp=23 fp=11 fn=19 precision=0.6765 recall=0.5476
- 轮廓划伤: gt=84 tp=52 fp=30 fn=28 precision=0.6341 recall=0.6190
- 锡丝残留: gt=12 tp=12 fp=7 fn=0 precision=0.6316 recall=1.0000
- 锡尖: gt=27 tp=27 fp=5 fn=0 precision=0.8438 recall=1.0000
- 锡膏: gt=46 tp=31 fp=8 fn=15 precision=0.7949 recall=0.6739
