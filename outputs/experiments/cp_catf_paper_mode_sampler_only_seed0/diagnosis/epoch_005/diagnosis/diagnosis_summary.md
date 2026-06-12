# Validation Error Diagnosis

- Status: completed
- TP: 253
- FP: 80
- FN: 182
- Precision: 0.7598
- Recall: 0.5724

## Diagnosis Vector
- small_object_score: 0.4916
- low_contrast_score: 0.4291
- class_imbalance_score: 0.9853
- localization_score: 0.0158
- false_positive_score: 0.2402

## Issues
- small_object_low_recall severity=medium suggestions=tiling, object-aware-crop, scale, copy-paste, mild-geometry
- background_interference severity=medium suggestions=background-diversity, light-noise, illumination-jitter
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=38 fp=8 fn=0 precision=0.8261 recall=1.0000
- OK3: gt=102 tp=100 fp=22 fn=0 precision=0.8197 recall=0.9804
- 加强筋打伤: gt=6 tp=0 fp=0 fn=6 precision=0.0000 recall=0.0000
- 开裂: gt=3 tp=0 fp=0 fn=3 precision=0.0000 recall=0.0000
- 油污: gt=17 tp=0 fp=0 fn=17 precision=0.0000 recall=0.0000
- 浅划伤: gt=14 tp=0 fp=0 fn=14 precision=0.0000 recall=0.0000
- 漏背锡: gt=32 tp=16 fp=21 fn=14 precision=0.4324 recall=0.5000
- 碰伤: gt=133 tp=80 fp=17 fn=52 precision=0.8247 recall=0.6015
- 脏污: gt=28 tp=0 fp=0 fn=28 precision=0.0000 recall=0.0000
- 轮廓划伤: gt=30 tp=16 fp=11 fn=12 precision=0.5926 recall=0.5333
- 锡丝残留: gt=6 tp=0 fp=0 fn=6 precision=0.0000 recall=0.0000
- 锡尖: gt=12 tp=2 fp=0 fn=10 precision=1.0000 recall=0.1667
- 锡膏: gt=21 tp=1 fp=1 fn=20 precision=0.5000 recall=0.0476
