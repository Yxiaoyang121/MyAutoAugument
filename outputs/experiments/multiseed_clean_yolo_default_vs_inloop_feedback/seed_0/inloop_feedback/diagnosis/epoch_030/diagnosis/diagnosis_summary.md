# Validation Error Diagnosis

- Status: completed
- TP: 713
- FP: 370
- FN: 172
- Precision: 0.6584
- Recall: 0.7878

## Diagnosis Vector
- small_object_score: 0.2716
- low_contrast_score: 0.5500
- class_imbalance_score: 0.8337
- localization_score: 0.0221
- false_positive_score: 0.3416

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=23 fn=0 precision=0.7356 recall=1.0000
- OK3: gt=274 tp=274 fp=69 fn=0 precision=0.7988 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=0 fn=1 precision=1.0000 recall=0.8750
- 开裂: gt=2 tp=2 fp=12 fn=0 precision=0.1429 recall=1.0000
- 油污: gt=18 tp=9 fp=55 fn=9 precision=0.1406 recall=0.5000
- 浅划伤: gt=13 tp=6 fp=0 fn=7 precision=1.0000 recall=0.4615
- 漏背锡: gt=46 tp=31 fp=37 fn=8 precision=0.4559 recall=0.6739
- 碰伤: gt=269 tp=185 fp=70 fn=79 precision=0.7255 recall=0.6877
- 脏污: gt=42 tp=23 fp=27 fn=17 precision=0.4600 recall=0.5476
- 轮廓划伤: gt=84 tp=44 fp=31 fn=36 precision=0.5867 recall=0.5238
- 锡丝残留: gt=12 tp=8 fp=7 fn=3 precision=0.5333 recall=0.6667
- 锡尖: gt=27 tp=24 fp=4 fn=2 precision=0.8571 recall=0.8889
- 锡膏: gt=46 tp=36 fp=35 fn=10 precision=0.5070 recall=0.7826
