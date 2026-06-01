# Validation Error Diagnosis

- Status: completed
- TP: 659
- FP: 254
- FN: 229
- Precision: 0.7218
- Recall: 0.7282

## Diagnosis Vector
- small_object_score: 0.3565
- low_contrast_score: 0.5397
- class_imbalance_score: 0.8786
- localization_score: 0.0188
- false_positive_score: 0.2782

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=23 fn=0 precision=0.7356 recall=1.0000
- OK3: gt=274 tp=268 fp=67 fn=6 precision=0.8000 recall=0.9781
- 加强筋打伤: gt=8 tp=5 fp=1 fn=3 precision=0.8333 recall=0.6250
- 开裂: gt=2 tp=2 fp=0 fn=0 precision=1.0000 recall=1.0000
- 油污: gt=18 tp=6 fp=14 fn=12 precision=0.3000 recall=0.3333
- 浅划伤: gt=13 tp=6 fp=4 fn=6 precision=0.6000 recall=0.4615
- 漏背锡: gt=46 tp=26 fp=13 fn=18 precision=0.6667 recall=0.5652
- 碰伤: gt=269 tp=146 fp=54 fn=118 precision=0.7300 recall=0.5428
- 脏污: gt=42 tp=18 fp=19 fn=19 precision=0.4865 recall=0.4286
- 轮廓划伤: gt=84 tp=54 fp=31 fn=26 precision=0.6353 recall=0.6429
- 锡丝残留: gt=12 tp=9 fp=6 fn=3 precision=0.6000 recall=0.7500
- 锡尖: gt=27 tp=23 fp=7 fn=4 precision=0.7667 recall=0.8519
- 锡膏: gt=46 tp=32 fp=15 fn=14 precision=0.6809 recall=0.6957
