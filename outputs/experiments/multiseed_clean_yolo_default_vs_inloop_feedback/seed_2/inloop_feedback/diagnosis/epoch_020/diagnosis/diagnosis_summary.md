# Validation Error Diagnosis

- Status: completed
- TP: 687
- FP: 297
- FN: 201
- Precision: 0.6982
- Recall: 0.7591

## Diagnosis Vector
- small_object_score: 0.2801
- low_contrast_score: 0.4853
- class_imbalance_score: 0.9369
- localization_score: 0.0188
- false_positive_score: 0.3018

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=24 fn=0 precision=0.7273 recall=1.0000
- OK3: gt=274 tp=274 fp=74 fn=0 precision=0.7874 recall=1.0000
- 加强筋打伤: gt=8 tp=2 fp=5 fn=5 precision=0.2857 recall=0.2500
- 开裂: gt=2 tp=2 fp=4 fn=0 precision=0.3333 recall=1.0000
- 油污: gt=18 tp=3 fp=22 fn=15 precision=0.1200 recall=0.1667
- 浅划伤: gt=13 tp=7 fp=3 fn=6 precision=0.7000 recall=0.5385
- 漏背锡: gt=46 tp=24 fp=15 fn=19 precision=0.6154 recall=0.5217
- 碰伤: gt=269 tp=181 fp=82 fn=82 precision=0.6882 recall=0.6729
- 脏污: gt=42 tp=15 fp=16 fn=26 precision=0.4839 recall=0.3571
- 轮廓划伤: gt=84 tp=55 fp=22 fn=23 precision=0.7143 recall=0.6548
- 锡丝残留: gt=12 tp=2 fp=4 fn=10 precision=0.3333 recall=0.1667
- 锡尖: gt=27 tp=24 fp=10 fn=3 precision=0.7059 recall=0.8889
- 锡膏: gt=46 tp=34 fp=16 fn=12 precision=0.6800 recall=0.7391
