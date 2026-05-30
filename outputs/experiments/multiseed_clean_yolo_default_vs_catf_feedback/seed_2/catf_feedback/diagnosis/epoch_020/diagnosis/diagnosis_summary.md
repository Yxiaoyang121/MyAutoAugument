# Validation Error Diagnosis

- Status: completed
- TP: 700
- FP: 349
- FN: 180
- Precision: 0.6673
- Recall: 0.7735

## Diagnosis Vector
- small_object_score: 0.2733
- low_contrast_score: 0.5575
- class_imbalance_score: 0.9953
- localization_score: 0.0276
- false_positive_score: 0.3327

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=24 fn=0 precision=0.7273 recall=1.0000
- OK3: gt=274 tp=274 fp=71 fn=0 precision=0.7942 recall=1.0000
- 加强筋打伤: gt=8 tp=5 fp=4 fn=3 precision=0.5556 recall=0.6250
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=5 fp=23 fn=13 precision=0.1786 recall=0.2778
- 浅划伤: gt=13 tp=7 fp=4 fn=6 precision=0.6364 recall=0.5385
- 漏背锡: gt=46 tp=26 fp=32 fn=7 precision=0.4483 recall=0.5652
- 碰伤: gt=269 tp=174 fp=100 fn=91 precision=0.6350 recall=0.6468
- 脏污: gt=42 tp=13 fp=12 fn=26 precision=0.5200 recall=0.3095
- 轮廓划伤: gt=84 tp=56 fp=21 fn=24 precision=0.7273 recall=0.6667
- 锡丝残留: gt=12 tp=11 fp=30 fn=0 precision=0.2683 recall=0.9167
- 锡尖: gt=27 tp=25 fp=11 fn=2 precision=0.6944 recall=0.9259
- 锡膏: gt=46 tp=40 fp=17 fn=6 precision=0.7018 recall=0.8696
