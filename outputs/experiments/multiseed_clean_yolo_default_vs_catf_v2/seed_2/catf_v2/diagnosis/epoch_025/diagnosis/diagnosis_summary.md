# Validation Error Diagnosis

- Status: completed
- TP: 719
- FP: 371
- FN: 157
- Precision: 0.6596
- Recall: 0.7945

## Diagnosis Vector
- small_object_score: 0.2530
- low_contrast_score: 0.4981
- class_imbalance_score: 0.8591
- localization_score: 0.0320
- false_positive_score: 0.3404

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=24 fn=0 precision=0.7273 recall=1.0000
- OK3: gt=274 tp=273 fp=69 fn=1 precision=0.7982 recall=0.9964
- 加强筋打伤: gt=8 tp=7 fp=1 fn=1 precision=0.8750 recall=0.8750
- 开裂: gt=2 tp=2 fp=6 fn=0 precision=0.2500 recall=1.0000
- 油污: gt=18 tp=7 fp=35 fn=11 precision=0.1667 recall=0.3889
- 浅划伤: gt=13 tp=6 fp=2 fn=6 precision=0.7500 recall=0.4615
- 漏背锡: gt=46 tp=32 fp=61 fn=6 precision=0.3441 recall=0.6957
- 碰伤: gt=269 tp=186 fp=102 fn=73 precision=0.6458 recall=0.6914
- 脏污: gt=42 tp=19 fp=6 fn=20 precision=0.7600 recall=0.4524
- 轮廓划伤: gt=84 tp=56 fp=33 fn=23 precision=0.6292 recall=0.6667
- 锡丝残留: gt=12 tp=9 fp=7 fn=3 precision=0.5625 recall=0.7500
- 锡尖: gt=27 tp=22 fp=7 fn=4 precision=0.7586 recall=0.8148
- 锡膏: gt=46 tp=36 fp=18 fn=9 precision=0.6667 recall=0.7826
