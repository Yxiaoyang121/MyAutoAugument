# Validation Error Diagnosis

- Status: completed
- TP: 725
- FP: 323
- FN: 170
- Precision: 0.6918
- Recall: 0.8011

## Diagnosis Vector
- small_object_score: 0.2479
- low_contrast_score: 0.5021
- class_imbalance_score: 0.9953
- localization_score: 0.0110
- false_positive_score: 0.3082

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=20 fn=0 precision=0.7619 recall=1.0000
- OK3: gt=274 tp=272 fp=70 fn=2 precision=0.7953 recall=0.9927
- 加强筋打伤: gt=8 tp=7 fp=0 fn=1 precision=1.0000 recall=0.8750
- 开裂: gt=2 tp=0 fp=13 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=6 fp=10 fn=11 precision=0.3750 recall=0.3333
- 浅划伤: gt=13 tp=8 fp=11 fn=5 precision=0.4211 recall=0.6154
- 漏背锡: gt=46 tp=30 fp=27 fn=16 precision=0.5263 recall=0.6522
- 碰伤: gt=269 tp=197 fp=83 fn=67 precision=0.7036 recall=0.7323
- 脏污: gt=42 tp=23 fp=29 fn=15 precision=0.4423 recall=0.5476
- 轮廓划伤: gt=84 tp=54 fp=30 fn=30 precision=0.6429 recall=0.6429
- 锡丝残留: gt=12 tp=8 fp=6 fn=4 precision=0.5714 recall=0.6667
- 锡尖: gt=27 tp=25 fp=8 fn=2 precision=0.7576 recall=0.9259
- 锡膏: gt=46 tp=31 fp=16 fn=15 precision=0.6596 recall=0.6739
