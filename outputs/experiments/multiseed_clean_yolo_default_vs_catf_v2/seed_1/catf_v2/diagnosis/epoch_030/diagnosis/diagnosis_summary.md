# Validation Error Diagnosis

- Status: completed
- TP: 712
- FP: 358
- FN: 173
- Precision: 0.6654
- Recall: 0.7867

## Diagnosis Vector
- small_object_score: 0.2666
- low_contrast_score: 0.5497
- class_imbalance_score: 0.9369
- localization_score: 0.0221
- false_positive_score: 0.3346

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=24 fn=0 precision=0.7273 recall=1.0000
- OK3: gt=274 tp=274 fp=74 fn=0 precision=0.7874 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=2 fn=1 precision=0.7778 recall=0.8750
- 开裂: gt=2 tp=1 fp=3 fn=1 precision=0.2500 recall=0.5000
- 油污: gt=18 tp=5 fp=21 fn=13 precision=0.1923 recall=0.2778
- 浅划伤: gt=13 tp=6 fp=8 fn=5 precision=0.4286 recall=0.4615
- 漏背锡: gt=46 tp=29 fp=41 fn=10 precision=0.4143 recall=0.6304
- 碰伤: gt=269 tp=177 fp=79 fn=86 precision=0.6914 recall=0.6580
- 脏污: gt=42 tp=30 fp=34 fn=11 precision=0.4688 recall=0.7143
- 轮廓划伤: gt=84 tp=57 fp=45 fn=25 precision=0.5588 recall=0.6786
- 锡丝残留: gt=12 tp=2 fp=1 fn=10 precision=0.6667 recall=0.1667
- 锡尖: gt=27 tp=25 fp=6 fn=2 precision=0.8065 recall=0.9259
- 锡膏: gt=46 tp=35 fp=20 fn=9 precision=0.6364 recall=0.7609
