# Validation Error Diagnosis

- Status: completed
- TP: 695
- FP: 319
- FN: 189
- Precision: 0.6854
- Recall: 0.7680

## Diagnosis Vector
- small_object_score: 0.3005
- low_contrast_score: 0.5905
- class_imbalance_score: 0.9175
- localization_score: 0.0232
- false_positive_score: 0.3146

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=21 fn=0 precision=0.7529 recall=1.0000
- OK3: gt=274 tp=274 fp=72 fn=0 precision=0.7919 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=1 fn=0 precision=0.8750 recall=0.8750
- 开裂: gt=2 tp=2 fp=1 fn=0 precision=0.6667 recall=1.0000
- 油污: gt=18 tp=4 fp=15 fn=14 precision=0.2105 recall=0.2222
- 浅划伤: gt=13 tp=6 fp=12 fn=5 precision=0.3333 recall=0.4615
- 漏背锡: gt=46 tp=32 fp=35 fn=10 precision=0.4776 recall=0.6957
- 碰伤: gt=269 tp=163 fp=59 fn=100 precision=0.7342 recall=0.6059
- 脏污: gt=42 tp=28 fp=31 fn=11 precision=0.4746 recall=0.6667
- 轮廓划伤: gt=84 tp=48 fp=44 fn=31 precision=0.5217 recall=0.5714
- 锡丝残留: gt=12 tp=7 fp=7 fn=5 precision=0.5000 recall=0.5833
- 锡尖: gt=27 tp=25 fp=4 fn=2 precision=0.8621 recall=0.9259
- 锡膏: gt=46 tp=35 fp=17 fn=11 precision=0.6731 recall=0.7609
