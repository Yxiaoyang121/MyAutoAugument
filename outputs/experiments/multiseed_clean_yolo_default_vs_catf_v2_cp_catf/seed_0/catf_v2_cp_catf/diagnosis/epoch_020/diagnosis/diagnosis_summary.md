# Validation Error Diagnosis

- Status: completed
- TP: 690
- FP: 368
- FN: 201
- Precision: 0.6522
- Recall: 0.7624

## Diagnosis Vector
- small_object_score: 0.2954
- low_contrast_score: 0.5236
- class_imbalance_score: 0.9953
- localization_score: 0.0155
- false_positive_score: 0.3478

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=24 fn=0 precision=0.7273 recall=1.0000
- OK3: gt=274 tp=274 fp=69 fn=0 precision=0.7988 recall=1.0000
- 加强筋打伤: gt=8 tp=6 fp=0 fn=2 precision=1.0000 recall=0.7500
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=6 fp=42 fn=11 precision=0.1250 recall=0.3333
- 浅划伤: gt=13 tp=8 fp=3 fn=5 precision=0.7273 recall=0.6154
- 漏背锡: gt=46 tp=31 fp=49 fn=11 precision=0.3875 recall=0.6739
- 碰伤: gt=269 tp=172 fp=70 fn=90 precision=0.7107 recall=0.6394
- 脏污: gt=42 tp=16 fp=14 fn=26 precision=0.5333 recall=0.3810
- 轮廓划伤: gt=84 tp=52 fp=48 fn=32 precision=0.5200 recall=0.6190
- 锡丝残留: gt=12 tp=7 fp=4 fn=5 precision=0.6364 recall=0.5833
- 锡尖: gt=27 tp=18 fp=6 fn=8 precision=0.7500 recall=0.6667
- 锡膏: gt=46 tp=36 fp=39 fn=9 precision=0.4800 recall=0.7826
