# Validation Error Diagnosis

- Status: completed
- TP: 659
- FP: 135
- FN: 231
- Precision: 0.8300
- Recall: 0.7282

## Diagnosis Vector
- small_object_score: 0.3413
- low_contrast_score: 0.5580
- class_imbalance_score: 0.8980
- localization_score: 0.0166
- false_positive_score: 0.1700

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=1 fn=0 precision=0.9846 recall=1.0000
- OK3: gt=274 tp=267 fp=6 fn=7 precision=0.9780 recall=0.9745
- 加强筋打伤: gt=8 tp=5 fp=1 fn=3 precision=0.8333 recall=0.6250
- 开裂: gt=2 tp=2 fp=0 fn=0 precision=1.0000 recall=1.0000
- 油污: gt=18 tp=5 fp=9 fn=13 precision=0.3571 recall=0.2778
- 浅划伤: gt=13 tp=8 fp=12 fn=5 precision=0.4000 recall=0.6154
- 漏背锡: gt=46 tp=29 fp=24 fn=14 precision=0.5472 recall=0.6304
- 碰伤: gt=269 tp=152 fp=40 fn=110 precision=0.7917 recall=0.5651
- 脏污: gt=42 tp=14 fp=7 fn=27 precision=0.6667 recall=0.3333
- 轮廓划伤: gt=84 tp=48 fp=10 fn=33 precision=0.8276 recall=0.5714
- 锡丝残留: gt=12 tp=6 fp=0 fn=6 precision=1.0000 recall=0.5000
- 锡尖: gt=27 tp=27 fp=6 fn=0 precision=0.8182 recall=1.0000
- 锡膏: gt=46 tp=32 fp=19 fn=13 precision=0.6275 recall=0.6957
