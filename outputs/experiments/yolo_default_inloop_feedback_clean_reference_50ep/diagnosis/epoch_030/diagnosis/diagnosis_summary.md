# Validation Error Diagnosis

- Status: completed
- TP: 696
- FP: 347
- FN: 183
- Precision: 0.6673
- Recall: 0.7691

## Diagnosis Vector
- small_object_score: 0.2971
- low_contrast_score: 0.5740
- class_imbalance_score: 0.8786
- localization_score: 0.0287
- false_positive_score: 0.3327

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
- 开裂: gt=2 tp=1 fp=6 fn=1 precision=0.1429 recall=0.5000
- 油污: gt=18 tp=6 fp=41 fn=11 precision=0.1277 recall=0.3333
- 浅划伤: gt=13 tp=6 fp=3 fn=7 precision=0.6667 recall=0.4615
- 漏背锡: gt=46 tp=29 fp=22 fn=11 precision=0.5686 recall=0.6304
- 碰伤: gt=269 tp=170 fp=96 fn=92 precision=0.6391 recall=0.6320
- 脏污: gt=42 tp=26 fp=15 fn=12 precision=0.6341 recall=0.6190
- 轮廓划伤: gt=84 tp=44 fp=43 fn=33 precision=0.5057 recall=0.5238
- 锡丝残留: gt=12 tp=8 fp=8 fn=3 precision=0.5000 recall=0.6667
- 锡尖: gt=27 tp=27 fp=7 fn=0 precision=0.7941 recall=1.0000
- 锡膏: gt=46 tp=34 fp=14 fn=12 precision=0.7083 recall=0.7391
