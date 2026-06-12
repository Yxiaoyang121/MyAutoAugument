# Validation Error Diagnosis

- Status: completed
- TP: 381
- FP: 111
- FN: 58
- Precision: 0.7744
- Recall: 0.8620

## Diagnosis Vector
- small_object_score: 0.1672
- low_contrast_score: 0.5414
- class_imbalance_score: 0.8853
- localization_score: 0.0068
- false_positive_score: 0.2256

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=38 fp=3 fn=0 precision=0.9268 recall=1.0000
- OK3: gt=102 tp=102 fp=25 fn=0 precision=0.8031 recall=1.0000
- 加强筋打伤: gt=6 tp=6 fp=2 fn=0 precision=0.7500 recall=1.0000
- 开裂: gt=3 tp=3 fp=2 fn=0 precision=0.6000 recall=1.0000
- 油污: gt=17 tp=8 fp=10 fn=9 precision=0.4444 recall=0.4706
- 浅划伤: gt=14 tp=4 fp=8 fn=9 precision=0.3333 recall=0.2857
- 漏背锡: gt=32 tp=26 fp=11 fn=4 precision=0.7027 recall=0.8125
- 碰伤: gt=133 tp=114 fp=17 fn=19 precision=0.8702 recall=0.8571
- 脏污: gt=28 tp=15 fp=16 fn=13 precision=0.4839 recall=0.5357
- 轮廓划伤: gt=30 tp=27 fp=4 fn=3 precision=0.8710 recall=0.9000
- 锡丝残留: gt=6 tp=6 fp=2 fn=0 precision=0.7500 recall=1.0000
- 锡尖: gt=12 tp=12 fp=3 fn=0 precision=0.8000 recall=1.0000
- 锡膏: gt=21 tp=20 fp=8 fn=1 precision=0.7143 recall=0.9524
