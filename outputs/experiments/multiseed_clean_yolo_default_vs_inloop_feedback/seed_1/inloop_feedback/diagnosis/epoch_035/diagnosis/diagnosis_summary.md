# Validation Error Diagnosis

- Status: completed
- TP: 710
- FP: 321
- FN: 181
- Precision: 0.6887
- Recall: 0.7845

## Diagnosis Vector
- small_object_score: 0.2801
- low_contrast_score: 0.5227
- class_imbalance_score: 0.9953
- localization_score: 0.0155
- false_positive_score: 0.3113

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=21 fn=0 precision=0.7529 recall=1.0000
- OK3: gt=274 tp=272 fp=73 fn=2 precision=0.7884 recall=0.9927
- 加强筋打伤: gt=8 tp=7 fp=0 fn=0 precision=1.0000 recall=0.8750
- 开裂: gt=2 tp=0 fp=1 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=8 fp=22 fn=10 precision=0.2667 recall=0.4444
- 浅划伤: gt=13 tp=6 fp=18 fn=6 precision=0.2500 recall=0.4615
- 漏背锡: gt=46 tp=35 fp=27 fn=7 precision=0.5645 recall=0.7609
- 碰伤: gt=269 tp=183 fp=85 fn=80 precision=0.6828 recall=0.6803
- 脏污: gt=42 tp=20 fp=19 fn=22 precision=0.5128 recall=0.4762
- 轮廓划伤: gt=84 tp=54 fp=35 fn=28 precision=0.6067 recall=0.6429
- 锡丝残留: gt=12 tp=10 fp=8 fn=2 precision=0.5556 recall=0.8333
- 锡尖: gt=27 tp=22 fp=3 fn=5 precision=0.8800 recall=0.8148
- 锡膏: gt=46 tp=29 fp=9 fn=17 precision=0.7632 recall=0.6304
