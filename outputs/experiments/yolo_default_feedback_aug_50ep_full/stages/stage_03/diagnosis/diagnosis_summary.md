# Validation Error Diagnosis

- Status: completed
- TP: 706
- FP: 404
- FN: 179
- Precision: 0.6360
- Recall: 0.7801

## Diagnosis Vector
- small_object_score: 0.2937
- low_contrast_score: 0.5506
- class_imbalance_score: 0.9953
- localization_score: 0.0221
- false_positive_score: 0.3640

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=23 fn=0 precision=0.7356 recall=1.0000
- OK3: gt=274 tp=271 fp=64 fn=3 precision=0.8090 recall=0.9891
- 加强筋打伤: gt=8 tp=7 fp=2 fn=1 precision=0.7778 recall=0.8750
- 开裂: gt=2 tp=0 fp=11 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=6 fp=34 fn=12 precision=0.1500 recall=0.3333
- 浅划伤: gt=13 tp=6 fp=3 fn=7 precision=0.6667 recall=0.4615
- 漏背锡: gt=46 tp=33 fp=27 fn=10 precision=0.5500 recall=0.7174
- 碰伤: gt=269 tp=177 fp=65 fn=86 precision=0.7314 recall=0.6580
- 脏污: gt=42 tp=28 fp=82 fn=9 precision=0.2545 recall=0.6667
- 轮廓划伤: gt=84 tp=49 fp=66 fn=29 precision=0.4261 recall=0.5833
- 锡丝残留: gt=12 tp=8 fp=12 fn=4 precision=0.4000 recall=0.6667
- 锡尖: gt=27 tp=25 fp=5 fn=2 precision=0.8333 recall=0.9259
- 锡膏: gt=46 tp=32 fp=10 fn=14 precision=0.7619 recall=0.6957
