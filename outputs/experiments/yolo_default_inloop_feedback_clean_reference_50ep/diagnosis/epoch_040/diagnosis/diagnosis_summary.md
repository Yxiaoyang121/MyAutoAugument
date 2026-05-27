# Validation Error Diagnosis

- Status: completed
- TP: 733
- FP: 315
- FN: 155
- Precision: 0.6994
- Recall: 0.8099

## Diagnosis Vector
- small_object_score: 0.2547
- low_contrast_score: 0.5352
- class_imbalance_score: 0.8337
- localization_score: 0.0188
- false_positive_score: 0.3006

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=20 fn=0 precision=0.7619 recall=1.0000
- OK3: gt=274 tp=272 fp=70 fn=2 precision=0.7953 recall=0.9927
- 加强筋打伤: gt=8 tp=7 fp=1 fn=1 precision=0.8750 recall=0.8750
- 开裂: gt=2 tp=2 fp=8 fn=0 precision=0.2000 recall=1.0000
- 油污: gt=18 tp=9 fp=17 fn=9 precision=0.3462 recall=0.5000
- 浅划伤: gt=13 tp=6 fp=5 fn=7 precision=0.5455 recall=0.4615
- 漏背锡: gt=46 tp=37 fp=30 fn=6 precision=0.5522 recall=0.8043
- 碰伤: gt=269 tp=201 fp=92 fn=62 precision=0.6860 recall=0.7472
- 脏污: gt=42 tp=26 fp=22 fn=14 precision=0.5417 recall=0.6190
- 轮廓划伤: gt=84 tp=43 fp=23 fn=35 precision=0.6515 recall=0.5119
- 锡丝残留: gt=12 tp=11 fp=10 fn=1 precision=0.5238 recall=0.9167
- 锡尖: gt=27 tp=24 fp=6 fn=3 precision=0.8000 recall=0.8889
- 锡膏: gt=46 tp=31 fp=11 fn=15 precision=0.7381 recall=0.6739
