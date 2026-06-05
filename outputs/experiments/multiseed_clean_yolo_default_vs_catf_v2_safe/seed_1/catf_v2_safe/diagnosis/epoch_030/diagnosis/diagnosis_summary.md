# Validation Error Diagnosis

- Status: completed
- TP: 686
- FP: 355
- FN: 197
- Precision: 0.6590
- Recall: 0.7580

## Diagnosis Vector
- small_object_score: 0.3107
- low_contrast_score: 0.5576
- class_imbalance_score: 0.8786
- localization_score: 0.0243
- false_positive_score: 0.3410

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=22 fn=0 precision=0.7442 recall=1.0000
- OK3: gt=274 tp=274 fp=74 fn=0 precision=0.7874 recall=1.0000
- 加强筋打伤: gt=8 tp=8 fp=6 fn=0 precision=0.5714 recall=1.0000
- 开裂: gt=2 tp=2 fp=4 fn=0 precision=0.3333 recall=1.0000
- 油污: gt=18 tp=8 fp=33 fn=10 precision=0.1951 recall=0.4444
- 浅划伤: gt=13 tp=6 fp=3 fn=5 precision=0.6667 recall=0.4615
- 漏背锡: gt=46 tp=30 fp=40 fn=10 precision=0.4286 recall=0.6522
- 碰伤: gt=269 tp=169 fp=93 fn=96 precision=0.6450 recall=0.6283
- 脏污: gt=42 tp=23 fp=24 fn=15 precision=0.4894 recall=0.5476
- 轮廓划伤: gt=84 tp=45 fp=34 fn=33 precision=0.5696 recall=0.5357
- 锡丝残留: gt=12 tp=4 fp=7 fn=8 precision=0.3636 recall=0.3333
- 锡尖: gt=27 tp=23 fp=5 fn=4 precision=0.8214 recall=0.8519
- 锡膏: gt=46 tp=30 fp=10 fn=16 precision=0.7500 recall=0.6522
