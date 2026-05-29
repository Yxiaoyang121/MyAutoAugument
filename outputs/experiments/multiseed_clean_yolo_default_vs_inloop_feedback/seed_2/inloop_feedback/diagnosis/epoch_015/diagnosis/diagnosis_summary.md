# Validation Error Diagnosis

- Status: completed
- TP: 660
- FP: 296
- FN: 226
- Precision: 0.6904
- Recall: 0.7293

## Diagnosis Vector
- small_object_score: 0.3226
- low_contrast_score: 0.5084
- class_imbalance_score: 0.9953
- localization_score: 0.0210
- false_positive_score: 0.3096

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=20 fn=0 precision=0.7619 recall=1.0000
- OK3: gt=274 tp=272 fp=62 fn=2 precision=0.8144 recall=0.9927
- 加强筋打伤: gt=8 tp=7 fp=2 fn=1 precision=0.7778 recall=0.8750
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=2 fp=19 fn=16 precision=0.0952 recall=0.1111
- 浅划伤: gt=13 tp=0 fp=0 fn=13 precision=0.0000 recall=0.0000
- 漏背锡: gt=46 tp=30 fp=22 fn=15 precision=0.5769 recall=0.6522
- 碰伤: gt=269 tp=171 fp=93 fn=92 precision=0.6477 recall=0.6357
- 脏污: gt=42 tp=14 fp=18 fn=24 precision=0.4375 recall=0.3333
- 轮廓划伤: gt=84 tp=42 fp=33 fn=34 precision=0.5600 recall=0.5000
- 锡丝残留: gt=12 tp=0 fp=0 fn=12 precision=0.0000 recall=0.0000
- 锡尖: gt=27 tp=25 fp=6 fn=2 precision=0.8065 recall=0.9259
- 锡膏: gt=46 tp=33 fp=21 fn=13 precision=0.6111 recall=0.7174
