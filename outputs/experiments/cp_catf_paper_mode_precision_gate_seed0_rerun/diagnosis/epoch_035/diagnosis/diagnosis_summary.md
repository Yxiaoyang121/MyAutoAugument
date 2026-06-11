# Validation Error Diagnosis

- Status: completed
- TP: 362
- FP: 113
- FN: 70
- Precision: 0.7621
- Recall: 0.8190

## Diagnosis Vector
- small_object_score: 0.2207
- low_contrast_score: 0.5514
- class_imbalance_score: 0.9442
- localization_score: 0.0226
- false_positive_score: 0.2379

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=38 fp=2 fn=0 precision=0.9500 recall=1.0000
- OK3: gt=102 tp=102 fp=22 fn=0 precision=0.8226 recall=1.0000
- 加强筋打伤: gt=6 tp=6 fp=0 fn=0 precision=1.0000 recall=1.0000
- 开裂: gt=3 tp=3 fp=6 fn=0 precision=0.3333 recall=1.0000
- 油污: gt=17 tp=2 fp=3 fn=13 precision=0.4000 recall=0.1176
- 浅划伤: gt=14 tp=6 fp=8 fn=5 precision=0.4286 recall=0.4286
- 漏背锡: gt=32 tp=26 fp=12 fn=6 precision=0.6842 recall=0.8125
- 碰伤: gt=133 tp=104 fp=21 fn=28 precision=0.8320 recall=0.7820
- 脏污: gt=28 tp=14 fp=26 fn=11 precision=0.3500 recall=0.5000
- 轮廓划伤: gt=30 tp=29 fp=6 fn=0 precision=0.8286 recall=0.9667
- 锡丝残留: gt=6 tp=4 fp=2 fn=2 precision=0.6667 recall=0.6667
- 锡尖: gt=12 tp=9 fp=3 fn=3 precision=0.7500 recall=0.7500
- 锡膏: gt=21 tp=19 fp=2 fn=2 precision=0.9048 recall=0.9048
