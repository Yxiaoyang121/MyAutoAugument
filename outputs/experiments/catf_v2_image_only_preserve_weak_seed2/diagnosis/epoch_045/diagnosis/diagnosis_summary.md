# Validation Error Diagnosis

- Status: completed
- TP: 724
- FP: 293
- FN: 161
- Precision: 0.7119
- Recall: 0.8000

## Diagnosis Vector
- small_object_score: 0.2530
- low_contrast_score: 0.5317
- class_imbalance_score: 0.9175
- localization_score: 0.0221
- false_positive_score: 0.2881

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=20 fn=0 precision=0.7619 recall=1.0000
- OK3: gt=274 tp=274 fp=65 fn=0 precision=0.8083 recall=1.0000
- 加强筋打伤: gt=8 tp=6 fp=2 fn=1 precision=0.7500 recall=0.7500
- 开裂: gt=2 tp=2 fp=10 fn=0 precision=0.1667 recall=1.0000
- 油污: gt=18 tp=4 fp=20 fn=13 precision=0.1667 recall=0.2222
- 浅划伤: gt=13 tp=8 fp=18 fn=5 precision=0.3077 recall=0.6154
- 漏背锡: gt=46 tp=31 fp=18 fn=12 precision=0.6327 recall=0.6739
- 碰伤: gt=269 tp=186 fp=81 fn=75 precision=0.6966 recall=0.6914
- 脏污: gt=42 tp=30 fp=18 fn=9 precision=0.6250 recall=0.7143
- 轮廓划伤: gt=84 tp=57 fp=29 fn=23 precision=0.6628 recall=0.6786
- 锡丝残留: gt=12 tp=9 fp=6 fn=3 precision=0.6000 recall=0.7500
- 锡尖: gt=27 tp=23 fp=2 fn=4 precision=0.9200 recall=0.8519
- 锡膏: gt=46 tp=30 fp=4 fn=16 precision=0.8824 recall=0.6522
