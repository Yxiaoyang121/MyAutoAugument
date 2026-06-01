# Validation Error Diagnosis

- Status: completed
- TP: 726
- FP: 292
- FN: 161
- Precision: 0.7132
- Recall: 0.8022

## Diagnosis Vector
- small_object_score: 0.2479
- low_contrast_score: 0.5248
- class_imbalance_score: 0.8980
- localization_score: 0.0199
- false_positive_score: 0.2868

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=19 fn=0 precision=0.7711 recall=1.0000
- OK3: gt=274 tp=273 fp=68 fn=1 precision=0.8006 recall=0.9964
- 加强筋打伤: gt=8 tp=3 fp=0 fn=5 precision=1.0000 recall=0.3750
- 开裂: gt=2 tp=2 fp=3 fn=0 precision=0.4000 recall=1.0000
- 油污: gt=18 tp=5 fp=12 fn=13 precision=0.2941 recall=0.2778
- 浅划伤: gt=13 tp=8 fp=10 fn=4 precision=0.4444 recall=0.6154
- 漏背锡: gt=46 tp=35 fp=24 fn=8 precision=0.5932 recall=0.7609
- 碰伤: gt=269 tp=193 fp=88 fn=70 precision=0.6868 recall=0.7175
- 脏污: gt=42 tp=25 fp=20 fn=12 precision=0.5556 recall=0.5952
- 轮廓划伤: gt=84 tp=55 fp=24 fn=26 precision=0.6962 recall=0.6548
- 锡丝残留: gt=12 tp=9 fp=6 fn=3 precision=0.6000 recall=0.7500
- 锡尖: gt=27 tp=24 fp=6 fn=3 precision=0.8000 recall=0.8889
- 锡膏: gt=46 tp=30 fp=12 fn=16 precision=0.7143 recall=0.6522
