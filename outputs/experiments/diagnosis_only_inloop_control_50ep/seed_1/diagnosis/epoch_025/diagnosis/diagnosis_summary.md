# Validation Error Diagnosis

- Status: completed
- TP: 660
- FP: 232
- FN: 236
- Precision: 0.7399
- Recall: 0.7293

## Diagnosis Vector
- small_object_score: 0.3328
- low_contrast_score: 0.5155
- class_imbalance_score: 0.9953
- localization_score: 0.0099
- false_positive_score: 0.2601

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=22 fn=0 precision=0.7442 recall=1.0000
- OK3: gt=274 tp=273 fp=62 fn=1 precision=0.8149 recall=0.9964
- 加强筋打伤: gt=8 tp=7 fp=3 fn=1 precision=0.7000 recall=0.8750
- 开裂: gt=2 tp=0 fp=1 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=5 fp=14 fn=13 precision=0.2632 recall=0.2778
- 浅划伤: gt=13 tp=4 fp=0 fn=9 precision=1.0000 recall=0.3077
- 漏背锡: gt=46 tp=29 fp=29 fn=14 precision=0.5000 recall=0.6304
- 碰伤: gt=269 tp=172 fp=50 fn=92 precision=0.7748 recall=0.6394
- 脏污: gt=42 tp=15 fp=15 fn=27 precision=0.5000 recall=0.3571
- 轮廓划伤: gt=84 tp=37 fp=22 fn=46 precision=0.6271 recall=0.4405
- 锡丝残留: gt=12 tp=4 fp=7 fn=8 precision=0.3636 recall=0.3333
- 锡尖: gt=27 tp=25 fp=5 fn=2 precision=0.8333 recall=0.9259
- 锡膏: gt=46 tp=25 fp=2 fn=21 precision=0.9259 recall=0.5435
