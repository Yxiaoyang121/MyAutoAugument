# Validation Error Diagnosis

- Status: completed
- TP: 698
- FP: 293
- FN: 185
- Precision: 0.7043
- Recall: 0.7713

## Diagnosis Vector
- small_object_score: 0.2869
- low_contrast_score: 0.5054
- class_imbalance_score: 0.8876
- localization_score: 0.0243
- false_positive_score: 0.2957

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=24 fn=0 precision=0.7273 recall=1.0000
- OK3: gt=274 tp=274 fp=67 fn=0 precision=0.8035 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=4 fn=1 precision=0.6364 recall=0.8750
- 开裂: gt=2 tp=2 fp=13 fn=0 precision=0.1333 recall=1.0000
- 油污: gt=18 tp=6 fp=19 fn=10 precision=0.2400 recall=0.3333
- 浅划伤: gt=13 tp=4 fp=3 fn=9 precision=0.5714 recall=0.3077
- 漏背锡: gt=46 tp=26 fp=10 fn=18 precision=0.7222 recall=0.5652
- 碰伤: gt=269 tp=175 fp=61 fn=91 precision=0.7415 recall=0.6506
- 脏污: gt=42 tp=23 fp=26 fn=12 precision=0.4694 recall=0.5476
- 轮廓划伤: gt=84 tp=53 fp=40 fn=25 precision=0.5699 recall=0.6310
- 锡丝残留: gt=12 tp=9 fp=3 fn=3 precision=0.7500 recall=0.7500
- 锡尖: gt=27 tp=22 fp=2 fn=5 precision=0.9167 recall=0.8148
- 锡膏: gt=46 tp=33 fp=21 fn=11 precision=0.6111 recall=0.7174
