# Validation Error Diagnosis

- Status: completed
- TP: 727
- FP: 290
- FN: 158
- Precision: 0.7148
- Recall: 0.8033

## Diagnosis Vector
- small_object_score: 0.2394
- low_contrast_score: 0.5187
- class_imbalance_score: 0.8980
- localization_score: 0.0221
- false_positive_score: 0.2852

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=21 fn=0 precision=0.7529 recall=1.0000
- OK3: gt=274 tp=274 fp=70 fn=0 precision=0.7965 recall=1.0000
- 加强筋打伤: gt=8 tp=6 fp=0 fn=1 precision=1.0000 recall=0.7500
- 开裂: gt=2 tp=2 fp=19 fn=0 precision=0.0952 recall=1.0000
- 油污: gt=18 tp=5 fp=12 fn=13 precision=0.2941 recall=0.2778
- 浅划伤: gt=13 tp=8 fp=7 fn=5 precision=0.5333 recall=0.6154
- 漏背锡: gt=46 tp=29 fp=19 fn=14 precision=0.6042 recall=0.6304
- 碰伤: gt=269 tp=198 fp=81 fn=65 precision=0.7097 recall=0.7361
- 脏污: gt=42 tp=26 fp=22 fn=12 precision=0.5417 recall=0.6190
- 轮廓划伤: gt=84 tp=49 fp=14 fn=31 precision=0.7778 recall=0.5833
- 锡丝残留: gt=12 tp=9 fp=5 fn=3 precision=0.6429 recall=0.7500
- 锡尖: gt=27 tp=23 fp=5 fn=4 precision=0.8214 recall=0.8519
- 锡膏: gt=46 tp=34 fp=15 fn=10 precision=0.6939 recall=0.7391
