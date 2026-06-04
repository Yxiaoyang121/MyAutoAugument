# Validation Error Diagnosis

- Status: completed
- TP: 671
- FP: 410
- FN: 211
- Precision: 0.6207
- Recall: 0.7414

## Diagnosis Vector
- small_object_score: 0.3260
- low_contrast_score: 0.5104
- class_imbalance_score: 0.9953
- localization_score: 0.0254
- false_positive_score: 0.3793

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=21 fn=0 precision=0.7529 recall=1.0000
- OK3: gt=274 tp=274 fp=69 fn=0 precision=0.7988 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=3 fn=1 precision=0.7000 recall=0.8750
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=4 fp=26 fn=14 precision=0.1333 recall=0.2222
- 浅划伤: gt=13 tp=6 fp=0 fn=7 precision=1.0000 recall=0.4615
- 漏背锡: gt=46 tp=29 fp=23 fn=13 precision=0.5577 recall=0.6304
- 碰伤: gt=269 tp=169 fp=113 fn=95 precision=0.5993 recall=0.6283
- 脏污: gt=42 tp=15 fp=102 fn=20 precision=0.1282 recall=0.3571
- 轮廓划伤: gt=84 tp=48 fp=34 fn=30 precision=0.5854 recall=0.5714
- 锡丝残留: gt=12 tp=3 fp=5 fn=9 precision=0.3750 recall=0.2500
- 锡尖: gt=27 tp=25 fp=8 fn=2 precision=0.7576 recall=0.9259
- 锡膏: gt=46 tp=27 fp=6 fn=18 precision=0.8182 recall=0.5870
