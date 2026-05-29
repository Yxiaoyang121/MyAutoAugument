# Validation Error Diagnosis

- Status: completed
- TP: 729
- FP: 473
- FN: 152
- Precision: 0.6065
- Recall: 0.8055

## Diagnosis Vector
- small_object_score: 0.2428
- low_contrast_score: 0.5132
- class_imbalance_score: 0.9953
- localization_score: 0.0265
- false_positive_score: 0.3935

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=27 fn=0 precision=0.7033 recall=1.0000
- OK3: gt=274 tp=274 fp=72 fn=0 precision=0.7919 recall=1.0000
- 加强筋打伤: gt=8 tp=5 fp=1 fn=3 precision=0.8333 recall=0.6250
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=6 fp=24 fn=12 precision=0.2000 recall=0.3333
- 浅划伤: gt=13 tp=6 fp=21 fn=5 precision=0.2222 recall=0.4615
- 漏背锡: gt=46 tp=25 fp=32 fn=17 precision=0.4386 recall=0.5435
- 碰伤: gt=269 tp=190 fp=128 fn=70 precision=0.5975 recall=0.7063
- 脏污: gt=42 tp=28 fp=50 fn=11 precision=0.3590 recall=0.6667
- 轮廓划伤: gt=84 tp=55 fp=63 fn=23 precision=0.4661 recall=0.6548
- 锡丝残留: gt=12 tp=10 fp=16 fn=2 precision=0.3846 recall=0.8333
- 锡尖: gt=27 tp=25 fp=10 fn=2 precision=0.7143 recall=0.9259
- 锡膏: gt=46 tp=41 fp=29 fn=5 precision=0.5857 recall=0.8913
