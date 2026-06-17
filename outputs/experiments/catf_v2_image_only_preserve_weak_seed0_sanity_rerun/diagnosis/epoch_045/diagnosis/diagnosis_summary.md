# Validation Error Diagnosis

- Status: completed
- TP: 706
- FP: 258
- FN: 183
- Precision: 0.7324
- Recall: 0.7801

## Diagnosis Vector
- small_object_score: 0.2852
- low_contrast_score: 0.5322
- class_imbalance_score: 0.8786
- localization_score: 0.0177
- false_positive_score: 0.2676

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=21 fn=0 precision=0.7529 recall=1.0000
- OK3: gt=274 tp=272 fp=63 fn=2 precision=0.8119 recall=0.9927
- 加强筋打伤: gt=8 tp=3 fp=1 fn=5 precision=0.7500 recall=0.3750
- 开裂: gt=2 tp=2 fp=12 fn=0 precision=0.1429 recall=1.0000
- 油污: gt=18 tp=6 fp=4 fn=12 precision=0.6000 recall=0.3333
- 浅划伤: gt=13 tp=6 fp=3 fn=7 precision=0.6667 recall=0.4615
- 漏背锡: gt=46 tp=32 fp=27 fn=10 precision=0.5424 recall=0.6957
- 碰伤: gt=269 tp=186 fp=68 fn=78 precision=0.7323 recall=0.6914
- 脏污: gt=42 tp=23 fp=11 fn=17 precision=0.6765 recall=0.5476
- 轮廓划伤: gt=84 tp=51 fp=26 fn=29 precision=0.6623 recall=0.6071
- 锡丝残留: gt=12 tp=9 fp=5 fn=3 precision=0.6429 recall=0.7500
- 锡尖: gt=27 tp=23 fp=4 fn=3 precision=0.8519 recall=0.8519
- 锡膏: gt=46 tp=29 fp=13 fn=17 precision=0.6905 recall=0.6304
