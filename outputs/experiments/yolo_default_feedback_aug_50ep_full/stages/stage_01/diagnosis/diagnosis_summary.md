# Validation Error Diagnosis

- Status: completed
- TP: 631
- FP: 346
- FN: 252
- Precision: 0.6459
- Recall: 0.6972

## Diagnosis Vector
- small_object_score: 0.3939
- low_contrast_score: 0.5502
- class_imbalance_score: 0.9953
- localization_score: 0.0243
- false_positive_score: 0.3541

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=29 fn=0 precision=0.6882 recall=1.0000
- OK3: gt=274 tp=270 fp=60 fn=3 precision=0.8182 recall=0.9854
- 加强筋打伤: gt=8 tp=0 fp=0 fn=8 precision=0.0000 recall=0.0000
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=4 fp=29 fn=13 precision=0.1212 recall=0.2222
- 浅划伤: gt=13 tp=5 fp=0 fn=7 precision=1.0000 recall=0.3846
- 漏背锡: gt=46 tp=34 fp=50 fn=7 precision=0.4048 recall=0.7391
- 碰伤: gt=269 tp=149 fp=59 fn=115 precision=0.7163 recall=0.5539
- 脏污: gt=42 tp=22 fp=67 fn=14 precision=0.2472 recall=0.5238
- 轮廓划伤: gt=84 tp=34 fp=27 fn=47 precision=0.5574 recall=0.4048
- 锡丝残留: gt=12 tp=0 fp=0 fn=12 precision=0.0000 recall=0.0000
- 锡尖: gt=27 tp=22 fp=5 fn=5 precision=0.8148 recall=0.8148
- 锡膏: gt=46 tp=27 fp=20 fn=19 precision=0.5745 recall=0.5870
