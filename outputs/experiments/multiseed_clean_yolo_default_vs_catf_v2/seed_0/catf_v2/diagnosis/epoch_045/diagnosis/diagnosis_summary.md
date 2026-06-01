# Validation Error Diagnosis

- Status: completed
- TP: 721
- FP: 272
- FN: 169
- Precision: 0.7261
- Recall: 0.7967

## Diagnosis Vector
- small_object_score: 0.2649
- low_contrast_score: 0.5402
- class_imbalance_score: 0.8786
- localization_score: 0.0166
- false_positive_score: 0.2739

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=20 fn=0 precision=0.7619 recall=1.0000
- OK3: gt=274 tp=274 fp=64 fn=0 precision=0.8107 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=0 fn=1 precision=1.0000 recall=0.8750
- 开裂: gt=2 tp=2 fp=6 fn=0 precision=0.2500 recall=1.0000
- 油污: gt=18 tp=6 fp=12 fn=10 precision=0.3333 recall=0.3333
- 浅划伤: gt=13 tp=8 fp=9 fn=5 precision=0.4706 recall=0.6154
- 漏背锡: gt=46 tp=33 fp=24 fn=10 precision=0.5789 recall=0.7174
- 碰伤: gt=269 tp=189 fp=69 fn=72 precision=0.7326 recall=0.7026
- 脏污: gt=42 tp=22 fp=13 fn=20 precision=0.6286 recall=0.5238
- 轮廓划伤: gt=84 tp=49 fp=33 fn=34 precision=0.5976 recall=0.5833
- 锡丝残留: gt=12 tp=10 fp=7 fn=2 precision=0.5882 recall=0.8333
- 锡尖: gt=27 tp=27 fp=3 fn=0 precision=0.9000 recall=1.0000
- 锡膏: gt=46 tp=30 fp=12 fn=15 precision=0.7143 recall=0.6522
