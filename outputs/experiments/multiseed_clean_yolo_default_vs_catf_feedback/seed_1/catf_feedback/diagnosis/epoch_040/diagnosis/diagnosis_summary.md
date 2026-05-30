# Validation Error Diagnosis

- Status: completed
- TP: 741
- FP: 328
- FN: 140
- Precision: 0.6932
- Recall: 0.8188

## Diagnosis Vector
- small_object_score: 0.2343
- low_contrast_score: 0.5629
- class_imbalance_score: 0.9175
- localization_score: 0.0265
- false_positive_score: 0.3068

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=24 fn=0 precision=0.7273 recall=1.0000
- OK3: gt=274 tp=274 fp=69 fn=0 precision=0.7988 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=1 fn=1 precision=0.8750 recall=0.8750
- 开裂: gt=2 tp=2 fp=8 fn=0 precision=0.2000 recall=1.0000
- 油污: gt=18 tp=4 fp=13 fn=14 precision=0.2353 recall=0.2222
- 浅划伤: gt=13 tp=8 fp=11 fn=5 precision=0.4211 recall=0.6154
- 漏背锡: gt=46 tp=34 fp=32 fn=5 precision=0.5152 recall=0.7391
- 碰伤: gt=269 tp=196 fp=89 fn=66 precision=0.6877 recall=0.7286
- 脏污: gt=42 tp=30 fp=18 fn=8 precision=0.6250 recall=0.7143
- 轮廓划伤: gt=84 tp=52 fp=24 fn=26 precision=0.6842 recall=0.6190
- 锡丝残留: gt=12 tp=11 fp=17 fn=1 precision=0.3929 recall=0.9167
- 锡尖: gt=27 tp=26 fp=9 fn=1 precision=0.7429 recall=0.9630
- 锡膏: gt=46 tp=33 fp=13 fn=13 precision=0.7174 recall=0.7174
