# Validation Error Diagnosis

- Status: completed
- TP: 694
- FP: 323
- FN: 185
- Precision: 0.6824
- Recall: 0.7669

## Diagnosis Vector
- small_object_score: 0.2971
- low_contrast_score: 0.5635
- class_imbalance_score: 0.9369
- localization_score: 0.0287
- false_positive_score: 0.3176

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=21 fn=0 precision=0.7529 recall=1.0000
- OK3: gt=274 tp=274 fp=70 fn=0 precision=0.7965 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=4 fn=1 precision=0.6364 recall=0.8750
- 开裂: gt=2 tp=2 fp=0 fn=0 precision=1.0000 recall=1.0000
- 油污: gt=18 tp=3 fp=22 fn=14 precision=0.1200 recall=0.1667
- 浅划伤: gt=13 tp=7 fp=4 fn=5 precision=0.6364 recall=0.5385
- 漏背锡: gt=46 tp=30 fp=41 fn=10 precision=0.4225 recall=0.6522
- 碰伤: gt=269 tp=169 fp=71 fn=93 precision=0.7042 recall=0.6283
- 脏污: gt=42 tp=25 fp=19 fn=13 precision=0.5682 recall=0.5952
- 轮廓划伤: gt=84 tp=48 fp=37 fn=29 precision=0.5647 recall=0.5714
- 锡丝残留: gt=12 tp=2 fp=5 fn=10 precision=0.2857 recall=0.1667
- 锡尖: gt=27 tp=26 fp=17 fn=1 precision=0.6047 recall=0.9630
- 锡膏: gt=46 tp=37 fp=12 fn=9 precision=0.7551 recall=0.8043
