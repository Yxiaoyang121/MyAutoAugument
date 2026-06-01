# Validation Error Diagnosis

- Status: completed
- TP: 696
- FP: 395
- FN: 186
- Precision: 0.6379
- Recall: 0.7691

## Diagnosis Vector
- small_object_score: 0.2937
- low_contrast_score: 0.5497
- class_imbalance_score: 0.9953
- localization_score: 0.0254
- false_positive_score: 0.3621

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=25 fn=0 precision=0.7191 recall=1.0000
- OK3: gt=274 tp=274 fp=71 fn=0 precision=0.7942 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=0 fn=1 precision=1.0000 recall=0.8750
- 开裂: gt=2 tp=0 fp=1 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=6 fp=43 fn=10 precision=0.1224 recall=0.3333
- 浅划伤: gt=13 tp=4 fp=1 fn=9 precision=0.8000 recall=0.3077
- 漏背锡: gt=46 tp=29 fp=36 fn=11 precision=0.4462 recall=0.6304
- 碰伤: gt=269 tp=175 fp=96 fn=90 precision=0.6458 recall=0.6506
- 脏污: gt=42 tp=19 fp=32 fn=20 precision=0.3725 recall=0.4524
- 轮廓划伤: gt=84 tp=49 fp=29 fn=29 precision=0.6282 recall=0.5833
- 锡丝残留: gt=12 tp=8 fp=11 fn=4 precision=0.4211 recall=0.6667
- 锡尖: gt=27 tp=25 fp=8 fn=2 precision=0.7576 recall=0.9259
- 锡膏: gt=46 tp=36 fp=42 fn=8 precision=0.4615 recall=0.7826
