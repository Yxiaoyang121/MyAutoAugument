# Validation Error Diagnosis

- Status: completed
- TP: 641
- FP: 260
- FN: 255
- Precision: 0.7114
- Recall: 0.7083

## Diagnosis Vector
- small_object_score: 0.3939
- low_contrast_score: 0.6080
- class_imbalance_score: 0.9953
- localization_score: 0.0099
- false_positive_score: 0.2886

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=25 fn=0 precision=0.7191 recall=1.0000
- OK3: gt=274 tp=264 fp=63 fn=10 precision=0.8073 recall=0.9635
- 加强筋打伤: gt=8 tp=7 fp=1 fn=1 precision=0.8750 recall=0.8750
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=4 fp=11 fn=14 precision=0.2667 recall=0.2222
- 浅划伤: gt=13 tp=7 fp=2 fn=6 precision=0.7778 recall=0.5385
- 漏背锡: gt=46 tp=35 fp=42 fn=7 precision=0.4545 recall=0.7609
- 碰伤: gt=269 tp=132 fp=24 fn=135 precision=0.8462 recall=0.4907
- 脏污: gt=42 tp=23 fp=44 fn=19 precision=0.3433 recall=0.5476
- 轮廓划伤: gt=84 tp=42 fp=21 fn=39 precision=0.6667 recall=0.5000
- 锡丝残留: gt=12 tp=10 fp=16 fn=2 precision=0.3846 recall=0.8333
- 锡尖: gt=27 tp=23 fp=7 fn=4 precision=0.7667 recall=0.8519
- 锡膏: gt=46 tp=30 fp=4 fn=16 precision=0.8824 recall=0.6522
