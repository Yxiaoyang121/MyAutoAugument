# Validation Error Diagnosis

- Status: completed
- TP: 652
- FP: 276
- FN: 239
- Precision: 0.7026
- Recall: 0.7204

## Diagnosis Vector
- small_object_score: 0.3633
- low_contrast_score: 0.5308
- class_imbalance_score: 0.9078
- localization_score: 0.0155
- false_positive_score: 0.2974

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=23 fn=0 precision=0.7356 recall=1.0000
- OK3: gt=274 tp=270 fp=65 fn=4 precision=0.8060 recall=0.9854
- 加强筋打伤: gt=8 tp=5 fp=0 fn=3 precision=1.0000 recall=0.6250
- 开裂: gt=2 tp=1 fp=0 fn=1 precision=1.0000 recall=0.5000
- 油污: gt=18 tp=5 fp=17 fn=13 precision=0.2273 recall=0.2778
- 浅划伤: gt=13 tp=7 fp=0 fn=6 precision=1.0000 recall=0.5385
- 漏背锡: gt=46 tp=34 fp=39 fn=7 precision=0.4658 recall=0.7391
- 碰伤: gt=269 tp=152 fp=58 fn=114 precision=0.7238 recall=0.5651
- 脏污: gt=42 tp=22 fp=22 fn=18 precision=0.5000 recall=0.5238
- 轮廓划伤: gt=84 tp=41 fp=33 fn=40 precision=0.5541 recall=0.4881
- 锡丝残留: gt=12 tp=3 fp=4 fn=9 precision=0.4286 recall=0.2500
- 锡尖: gt=27 tp=21 fp=8 fn=6 precision=0.7241 recall=0.7778
- 锡膏: gt=46 tp=27 fp=7 fn=18 precision=0.7941 recall=0.5870
