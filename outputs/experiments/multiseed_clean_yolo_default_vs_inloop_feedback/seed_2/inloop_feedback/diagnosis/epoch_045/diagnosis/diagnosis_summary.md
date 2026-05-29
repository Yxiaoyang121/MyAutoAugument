# Validation Error Diagnosis

- Status: completed
- TP: 715
- FP: 277
- FN: 166
- Precision: 0.7208
- Recall: 0.7901

## Diagnosis Vector
- small_object_score: 0.2615
- low_contrast_score: 0.5178
- class_imbalance_score: 0.9369
- localization_score: 0.0265
- false_positive_score: 0.2792

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=20 fn=0 precision=0.7619 recall=1.0000
- OK3: gt=274 tp=270 fp=65 fn=4 precision=0.8060 recall=0.9854
- 加强筋打伤: gt=8 tp=6 fp=0 fn=2 precision=1.0000 recall=0.7500
- 开裂: gt=2 tp=2 fp=4 fn=0 precision=0.3333 recall=1.0000
- 油污: gt=18 tp=3 fp=18 fn=15 precision=0.1429 recall=0.1667
- 浅划伤: gt=13 tp=6 fp=6 fn=7 precision=0.5000 recall=0.4615
- 漏背锡: gt=46 tp=33 fp=19 fn=10 precision=0.6346 recall=0.7174
- 碰伤: gt=269 tp=183 fp=71 fn=73 precision=0.7205 recall=0.6803
- 脏污: gt=42 tp=24 fp=15 fn=13 precision=0.6154 recall=0.5714
- 轮廓划伤: gt=84 tp=58 fp=32 fn=24 precision=0.6444 recall=0.6905
- 锡丝残留: gt=12 tp=7 fp=4 fn=5 precision=0.6364 recall=0.5833
- 锡尖: gt=27 tp=27 fp=3 fn=0 precision=0.9000 recall=1.0000
- 锡膏: gt=46 tp=32 fp=20 fn=13 precision=0.6154 recall=0.6957
