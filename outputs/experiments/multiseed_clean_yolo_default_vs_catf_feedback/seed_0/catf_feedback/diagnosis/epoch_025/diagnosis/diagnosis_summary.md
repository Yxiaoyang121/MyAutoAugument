# Validation Error Diagnosis

- Status: completed
- TP: 719
- FP: 377
- FN: 170
- Precision: 0.6560
- Recall: 0.7945

## Diagnosis Vector
- small_object_score: 0.2632
- low_contrast_score: 0.5538
- class_imbalance_score: 0.8786
- localization_score: 0.0177
- false_positive_score: 0.3440

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=26 fn=0 precision=0.7111 recall=1.0000
- OK3: gt=274 tp=274 fp=70 fn=0 precision=0.7965 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=3 fn=1 precision=0.7000 recall=0.8750
- 开裂: gt=2 tp=2 fp=7 fn=0 precision=0.2222 recall=1.0000
- 油污: gt=18 tp=6 fp=33 fn=11 precision=0.1538 recall=0.3333
- 浅划伤: gt=13 tp=6 fp=0 fn=7 precision=1.0000 recall=0.4615
- 漏背锡: gt=46 tp=30 fp=26 fn=11 precision=0.5357 recall=0.6522
- 碰伤: gt=269 tp=189 fp=93 fn=74 precision=0.6702 recall=0.7026
- 脏污: gt=42 tp=23 fp=40 fn=17 precision=0.3651 recall=0.5476
- 轮廓划伤: gt=84 tp=50 fp=44 fn=34 precision=0.5319 recall=0.5952
- 锡丝残留: gt=12 tp=8 fp=4 fn=4 precision=0.6667 recall=0.6667
- 锡尖: gt=27 tp=23 fp=2 fn=4 precision=0.9200 recall=0.8519
- 锡膏: gt=46 tp=37 fp=29 fn=7 precision=0.5606 recall=0.8043
