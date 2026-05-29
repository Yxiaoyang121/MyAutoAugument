# Validation Error Diagnosis

- Status: completed
- TP: 737
- FP: 296
- FN: 156
- Precision: 0.7135
- Recall: 0.8144

## Diagnosis Vector
- small_object_score: 0.2428
- low_contrast_score: 0.5705
- class_imbalance_score: 0.8397
- localization_score: 0.0133
- false_positive_score: 0.2865

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=21 fn=0 precision=0.7529 recall=1.0000
- OK3: gt=274 tp=272 fp=62 fn=2 precision=0.8144 recall=0.9927
- 加强筋打伤: gt=8 tp=7 fp=0 fn=1 precision=1.0000 recall=0.8750
- 开裂: gt=2 tp=2 fp=0 fn=0 precision=1.0000 recall=1.0000
- 油污: gt=18 tp=8 fp=26 fn=9 precision=0.2353 recall=0.4444
- 浅划伤: gt=13 tp=7 fp=8 fn=4 precision=0.4667 recall=0.5385
- 漏背锡: gt=46 tp=31 fp=23 fn=12 precision=0.5741 recall=0.6739
- 碰伤: gt=269 tp=189 fp=73 fn=77 precision=0.7214 recall=0.7026
- 脏污: gt=42 tp=33 fp=20 fn=9 precision=0.6226 recall=0.7857
- 轮廓划伤: gt=84 tp=53 fp=35 fn=28 precision=0.6023 recall=0.6310
- 锡丝残留: gt=12 tp=12 fp=8 fn=0 precision=0.6000 recall=1.0000
- 锡尖: gt=27 tp=25 fp=4 fn=2 precision=0.8621 recall=0.9259
- 锡膏: gt=46 tp=34 fp=16 fn=12 precision=0.6800 recall=0.7391
