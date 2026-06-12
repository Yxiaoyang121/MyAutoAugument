# Validation Error Diagnosis

- Status: completed
- TP: 226
- FP: 51
- FN: 214
- Precision: 0.8159
- Recall: 0.5113

## Diagnosis Vector
- small_object_score: 0.5518
- low_contrast_score: 0.5044
- class_imbalance_score: 0.9853
- localization_score: 0.0045
- false_positive_score: 0.1841

## Issues
- small_object_low_recall severity=medium suggestions=tiling, object-aware-crop, scale, copy-paste, mild-geometry
- background_interference severity=medium suggestions=background-diversity, light-noise, illumination-jitter
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=38 fp=2 fn=0 precision=0.9500 recall=1.0000
- OK3: gt=102 tp=100 fp=23 fn=2 precision=0.8130 recall=0.9804
- 加强筋打伤: gt=6 tp=0 fp=0 fn=6 precision=0.0000 recall=0.0000
- 开裂: gt=3 tp=0 fp=0 fn=3 precision=0.0000 recall=0.0000
- 油污: gt=17 tp=0 fp=0 fn=17 precision=0.0000 recall=0.0000
- 浅划伤: gt=14 tp=0 fp=0 fn=14 precision=0.0000 recall=0.0000
- 漏背锡: gt=32 tp=8 fp=2 fn=24 precision=0.8000 recall=0.2500
- 碰伤: gt=133 tp=60 fp=15 fn=71 precision=0.8000 recall=0.4511
- 脏污: gt=28 tp=0 fp=0 fn=28 precision=0.0000 recall=0.0000
- 轮廓划伤: gt=30 tp=5 fp=5 fn=25 precision=0.5000 recall=0.1667
- 锡丝残留: gt=6 tp=0 fp=0 fn=6 precision=0.0000 recall=0.0000
- 锡尖: gt=12 tp=5 fp=4 fn=7 precision=0.5556 recall=0.4167
- 锡膏: gt=21 tp=10 fp=0 fn=11 precision=1.0000 recall=0.4762
