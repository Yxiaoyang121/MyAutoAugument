# Validation Error Diagnosis

- Status: completed
- TP: 726
- FP: 296
- FN: 157
- Precision: 0.7104
- Recall: 0.8022

## Diagnosis Vector
- small_object_score: 0.2547
- low_contrast_score: 0.5487
- class_imbalance_score: 0.8786
- localization_score: 0.0243
- false_positive_score: 0.2896

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=22 fn=0 precision=0.7442 recall=1.0000
- OK3: gt=274 tp=274 fp=65 fn=0 precision=0.8083 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=1 fn=1 precision=0.8750 recall=0.8750
- 开裂: gt=2 tp=2 fp=5 fn=0 precision=0.2857 recall=1.0000
- 油污: gt=18 tp=6 fp=28 fn=12 precision=0.1765 recall=0.3333
- 浅划伤: gt=13 tp=7 fp=5 fn=5 precision=0.5833 recall=0.5385
- 漏背锡: gt=46 tp=35 fp=25 fn=4 precision=0.5833 recall=0.7609
- 碰伤: gt=269 tp=194 fp=67 fn=69 precision=0.7433 recall=0.7212
- 脏污: gt=42 tp=26 fp=13 fn=16 precision=0.6667 recall=0.6190
- 轮廓划伤: gt=84 tp=44 fp=43 fn=32 precision=0.5057 recall=0.5238
- 锡丝残留: gt=12 tp=12 fp=9 fn=0 precision=0.5714 recall=1.0000
- 锡尖: gt=27 tp=25 fp=5 fn=2 precision=0.8333 recall=0.9259
- 锡膏: gt=46 tp=30 fp=8 fn=16 precision=0.7895 recall=0.6522
