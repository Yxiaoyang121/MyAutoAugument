# Validation Error Diagnosis

- Status: completed
- TP: 735
- FP: 269
- FN: 149
- Precision: 0.7321
- Recall: 0.8122

## Diagnosis Vector
- small_object_score: 0.2513
- low_contrast_score: 0.5601
- class_imbalance_score: 0.8397
- localization_score: 0.0232
- false_positive_score: 0.2679

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=20 fn=0 precision=0.7619 recall=1.0000
- OK3: gt=274 tp=274 fp=67 fn=0 precision=0.8035 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=0 fn=1 precision=1.0000 recall=0.8750
- 开裂: gt=2 tp=2 fp=4 fn=0 precision=0.3333 recall=1.0000
- 油污: gt=18 tp=8 fp=22 fn=10 precision=0.2667 recall=0.4444
- 浅划伤: gt=13 tp=6 fp=4 fn=7 precision=0.6000 recall=0.4615
- 漏背锡: gt=46 tp=35 fp=18 fn=8 precision=0.6604 recall=0.7609
- 碰伤: gt=269 tp=191 fp=72 fn=72 precision=0.7262 recall=0.7100
- 脏污: gt=42 tp=28 fp=12 fn=10 precision=0.7000 recall=0.6667
- 轮廓划伤: gt=84 tp=52 fp=24 fn=28 precision=0.6842 recall=0.6190
- 锡丝残留: gt=12 tp=10 fp=6 fn=1 precision=0.6250 recall=0.8333
- 锡尖: gt=27 tp=27 fp=6 fn=0 precision=0.8182 recall=1.0000
- 锡膏: gt=46 tp=31 fp=14 fn=12 precision=0.6889 recall=0.6739
