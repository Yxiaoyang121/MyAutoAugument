# Validation Error Diagnosis

- Status: completed
- TP: 717
- FP: 314
- FN: 164
- Precision: 0.6954
- Recall: 0.7923

## Diagnosis Vector
- small_object_score: 0.2513
- low_contrast_score: 0.5250
- class_imbalance_score: 0.9953
- localization_score: 0.0265
- false_positive_score: 0.3046

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=24 fn=0 precision=0.7273 recall=1.0000
- OK3: gt=274 tp=273 fp=68 fn=1 precision=0.8006 recall=0.9964
- 加强筋打伤: gt=8 tp=8 fp=2 fn=0 precision=0.8000 recall=1.0000
- 开裂: gt=2 tp=0 fp=9 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=6 fp=17 fn=12 precision=0.2609 recall=0.3333
- 浅划伤: gt=13 tp=7 fp=7 fn=6 precision=0.5000 recall=0.5385
- 漏背锡: gt=46 tp=30 fp=33 fn=7 precision=0.4762 recall=0.6522
- 碰伤: gt=269 tp=194 fp=81 fn=71 precision=0.7055 recall=0.7212
- 脏污: gt=42 tp=25 fp=13 fn=14 precision=0.6579 recall=0.5952
- 轮廓划伤: gt=84 tp=44 fp=29 fn=32 precision=0.6027 recall=0.5238
- 锡丝残留: gt=12 tp=10 fp=9 fn=2 precision=0.5263 recall=0.8333
- 锡尖: gt=27 tp=25 fp=7 fn=2 precision=0.7812 recall=0.9259
- 锡膏: gt=46 tp=31 fp=15 fn=15 precision=0.6739 recall=0.6739
