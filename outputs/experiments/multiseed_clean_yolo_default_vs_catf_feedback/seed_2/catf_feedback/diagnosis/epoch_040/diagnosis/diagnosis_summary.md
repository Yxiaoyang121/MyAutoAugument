# Validation Error Diagnosis

- Status: completed
- TP: 742
- FP: 356
- FN: 145
- Precision: 0.6758
- Recall: 0.8199

## Diagnosis Vector
- small_object_score: 0.2360
- low_contrast_score: 0.5262
- class_imbalance_score: 0.9953
- localization_score: 0.0199
- false_positive_score: 0.3242

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=26 fn=0 precision=0.7111 recall=1.0000
- OK3: gt=274 tp=272 fp=70 fn=2 precision=0.7953 recall=0.9927
- 加强筋打伤: gt=8 tp=7 fp=3 fn=1 precision=0.7000 recall=0.8750
- 开裂: gt=2 tp=0 fp=7 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=5 fp=15 fn=13 precision=0.2500 recall=0.2778
- 浅划伤: gt=13 tp=8 fp=6 fn=5 precision=0.5714 recall=0.6154
- 漏背锡: gt=46 tp=36 fp=31 fn=6 precision=0.5373 recall=0.7826
- 碰伤: gt=269 tp=206 fp=111 fn=54 precision=0.6498 recall=0.7658
- 脏污: gt=42 tp=24 fp=32 fn=17 precision=0.4286 recall=0.5714
- 轮廓划伤: gt=84 tp=48 fp=24 fn=34 precision=0.6667 recall=0.5714
- 锡丝残留: gt=12 tp=12 fp=8 fn=0 precision=0.6000 recall=1.0000
- 锡尖: gt=27 tp=25 fp=6 fn=2 precision=0.8065 recall=0.9259
- 锡膏: gt=46 tp=35 fp=17 fn=9 precision=0.6731 recall=0.7609
