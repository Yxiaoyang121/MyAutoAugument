# Validation Error Diagnosis

- Status: completed
- TP: 737
- FP: 435
- FN: 143
- Precision: 0.6288
- Recall: 0.8144

## Diagnosis Vector
- small_object_score: 0.2377
- low_contrast_score: 0.5622
- class_imbalance_score: 0.9175
- localization_score: 0.0276
- false_positive_score: 0.3712

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=31 fn=0 precision=0.6737 recall=1.0000
- OK3: gt=274 tp=274 fp=66 fn=0 precision=0.8059 recall=1.0000
- 加强筋打伤: gt=8 tp=5 fp=0 fn=3 precision=1.0000 recall=0.6250
- 开裂: gt=2 tp=2 fp=7 fn=0 precision=0.2222 recall=1.0000
- 油污: gt=18 tp=4 fp=13 fn=12 precision=0.2353 recall=0.2222
- 浅划伤: gt=13 tp=10 fp=5 fn=3 precision=0.6667 recall=0.7692
- 漏背锡: gt=46 tp=38 fp=79 fn=0 precision=0.3248 recall=0.8261
- 碰伤: gt=269 tp=197 fp=113 fn=67 precision=0.6355 recall=0.7323
- 脏污: gt=42 tp=23 fp=18 fn=17 precision=0.5610 recall=0.5476
- 轮廓划伤: gt=84 tp=52 fp=59 fn=26 precision=0.4685 recall=0.6190
- 锡丝残留: gt=12 tp=9 fp=7 fn=3 precision=0.5625 recall=0.7500
- 锡尖: gt=27 tp=23 fp=3 fn=4 precision=0.8846 recall=0.8519
- 锡膏: gt=46 tp=36 fp=34 fn=8 precision=0.5143 recall=0.7826
