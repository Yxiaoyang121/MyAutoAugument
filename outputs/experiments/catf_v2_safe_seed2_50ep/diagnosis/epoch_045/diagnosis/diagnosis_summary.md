# Validation Error Diagnosis

- Status: completed
- TP: 728
- FP: 293
- FN: 157
- Precision: 0.7130
- Recall: 0.8044

## Diagnosis Vector
- small_object_score: 0.2513
- low_contrast_score: 0.5408
- class_imbalance_score: 0.8980
- localization_score: 0.0221
- false_positive_score: 0.2870

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=20 fn=0 precision=0.7619 recall=1.0000
- OK3: gt=274 tp=274 fp=63 fn=0 precision=0.8131 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=2 fn=1 precision=0.7778 recall=0.8750
- 开裂: gt=2 tp=2 fp=15 fn=0 precision=0.1176 recall=1.0000
- 油污: gt=18 tp=5 fp=22 fn=12 precision=0.1852 recall=0.2778
- 浅划伤: gt=13 tp=8 fp=16 fn=4 precision=0.3333 recall=0.6154
- 漏背锡: gt=46 tp=36 fp=18 fn=7 precision=0.6667 recall=0.7826
- 碰伤: gt=269 tp=185 fp=77 fn=77 precision=0.7061 recall=0.6877
- 脏污: gt=42 tp=25 fp=11 fn=13 precision=0.6944 recall=0.5952
- 轮廓划伤: gt=84 tp=57 fp=30 fn=23 precision=0.6552 recall=0.6786
- 锡丝残留: gt=12 tp=9 fp=6 fn=3 precision=0.6000 recall=0.7500
- 锡尖: gt=27 tp=23 fp=3 fn=4 precision=0.8846 recall=0.8519
- 锡膏: gt=46 tp=33 fp=10 fn=13 precision=0.7674 recall=0.7174
