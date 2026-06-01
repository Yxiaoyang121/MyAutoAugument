# Validation Error Diagnosis

- Status: completed
- TP: 714
- FP: 321
- FN: 174
- Precision: 0.6899
- Recall: 0.7890

## Diagnosis Vector
- small_object_score: 0.2564
- low_contrast_score: 0.5621
- class_imbalance_score: 0.8786
- localization_score: 0.0188
- false_positive_score: 0.3101

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=23 fn=0 precision=0.7356 recall=1.0000
- OK3: gt=274 tp=273 fp=69 fn=1 precision=0.7982 recall=0.9964
- 加强筋打伤: gt=8 tp=7 fp=1 fn=1 precision=0.8750 recall=0.8750
- 开裂: gt=2 tp=2 fp=3 fn=0 precision=0.4000 recall=1.0000
- 油污: gt=18 tp=6 fp=13 fn=12 precision=0.3158 recall=0.3333
- 浅划伤: gt=13 tp=8 fp=11 fn=5 precision=0.4211 recall=0.6154
- 漏背锡: gt=46 tp=32 fp=40 fn=6 precision=0.4444 recall=0.6957
- 碰伤: gt=269 tp=184 fp=90 fn=79 precision=0.6715 recall=0.6840
- 脏污: gt=42 tp=23 fp=20 fn=16 precision=0.5349 recall=0.5476
- 轮廓划伤: gt=84 tp=53 fp=21 fn=31 precision=0.7162 recall=0.6310
- 锡丝残留: gt=12 tp=10 fp=11 fn=2 precision=0.4762 recall=0.8333
- 锡尖: gt=27 tp=23 fp=6 fn=4 precision=0.7931 recall=0.8519
- 锡膏: gt=46 tp=29 fp=13 fn=17 precision=0.6905 recall=0.6304
