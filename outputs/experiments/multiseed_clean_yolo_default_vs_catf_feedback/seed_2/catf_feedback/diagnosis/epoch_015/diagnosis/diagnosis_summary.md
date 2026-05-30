# Validation Error Diagnosis

- Status: completed
- TP: 682
- FP: 333
- FN: 199
- Precision: 0.6719
- Recall: 0.7536

## Diagnosis Vector
- small_object_score: 0.3107
- low_contrast_score: 0.4990
- class_imbalance_score: 0.9953
- localization_score: 0.0265
- false_positive_score: 0.3281

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=25 fn=0 precision=0.7191 recall=1.0000
- OK3: gt=274 tp=274 fp=59 fn=0 precision=0.8228 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=2 fn=1 precision=0.7778 recall=0.8750
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=2 fp=36 fn=16 precision=0.0526 recall=0.1111
- 浅划伤: gt=13 tp=0 fp=0 fn=13 precision=0.0000 recall=0.0000
- 漏背锡: gt=46 tp=33 fp=20 fn=12 precision=0.6226 recall=0.7174
- 碰伤: gt=269 tp=189 fp=110 fn=71 precision=0.6321 recall=0.7026
- 脏污: gt=42 tp=18 fp=35 fn=19 precision=0.3396 recall=0.4286
- 轮廓划伤: gt=84 tp=35 fp=22 fn=40 precision=0.6140 recall=0.4167
- 锡丝残留: gt=12 tp=0 fp=0 fn=12 precision=0.0000 recall=0.0000
- 锡尖: gt=27 tp=25 fp=11 fn=2 precision=0.6944 recall=0.9259
- 锡膏: gt=46 tp=35 fp=13 fn=11 precision=0.7292 recall=0.7609
