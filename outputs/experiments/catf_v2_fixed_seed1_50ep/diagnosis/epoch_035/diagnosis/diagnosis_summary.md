# Validation Error Diagnosis

- Status: completed
- TP: 700
- FP: 272
- FN: 186
- Precision: 0.7202
- Recall: 0.7735

## Diagnosis Vector
- small_object_score: 0.2750
- low_contrast_score: 0.5022
- class_imbalance_score: 0.8980
- localization_score: 0.0210
- false_positive_score: 0.2798

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=20 fn=0 precision=0.7619 recall=1.0000
- OK3: gt=274 tp=273 fp=69 fn=1 precision=0.7982 recall=0.9964
- 加强筋打伤: gt=8 tp=7 fp=1 fn=1 precision=0.8750 recall=0.8750
- 开裂: gt=2 tp=2 fp=0 fn=0 precision=1.0000 recall=1.0000
- 油污: gt=18 tp=5 fp=12 fn=12 precision=0.2941 recall=0.2778
- 浅划伤: gt=13 tp=7 fp=5 fn=5 precision=0.5833 recall=0.5385
- 漏背锡: gt=46 tp=29 fp=24 fn=10 precision=0.5472 recall=0.6304
- 碰伤: gt=269 tp=190 fp=79 fn=70 precision=0.7063 recall=0.7063
- 脏污: gt=42 tp=20 fp=11 fn=22 precision=0.6452 recall=0.4762
- 轮廓划伤: gt=84 tp=49 fp=41 fn=34 precision=0.5444 recall=0.5833
- 锡丝残留: gt=12 tp=8 fp=6 fn=4 precision=0.5714 recall=0.6667
- 锡尖: gt=27 tp=21 fp=3 fn=6 precision=0.8750 recall=0.7778
- 锡膏: gt=46 tp=25 fp=1 fn=21 precision=0.9615 recall=0.5435
