# Validation Error Diagnosis

- Status: completed
- TP: 287
- FP: 95
- FN: 151
- Precision: 0.7513
- Recall: 0.6493

## Diagnosis Vector
- small_object_score: 0.3779
- low_contrast_score: 0.4066
- class_imbalance_score: 0.9853
- localization_score: 0.0090
- false_positive_score: 0.2487

## Issues
- background_interference severity=medium suggestions=background-diversity, light-noise, illumination-jitter
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=38 fp=1 fn=0 precision=0.9744 recall=1.0000
- OK3: gt=102 tp=100 fp=29 fn=2 precision=0.7752 recall=0.9804
- 加强筋打伤: gt=6 tp=0 fp=0 fn=6 precision=0.0000 recall=0.0000
- 开裂: gt=3 tp=0 fp=0 fn=3 precision=0.0000 recall=0.0000
- 油污: gt=17 tp=0 fp=5 fn=17 precision=0.0000 recall=0.0000
- 浅划伤: gt=14 tp=0 fp=0 fn=14 precision=0.0000 recall=0.0000
- 漏背锡: gt=32 tp=2 fp=0 fn=29 precision=1.0000 recall=0.0625
- 碰伤: gt=133 tp=95 fp=36 fn=37 precision=0.7252 recall=0.7143
- 脏污: gt=28 tp=0 fp=0 fn=28 precision=0.0000 recall=0.0000
- 轮廓划伤: gt=30 tp=27 fp=10 fn=3 precision=0.7297 recall=0.9000
- 锡丝残留: gt=6 tp=0 fp=0 fn=6 precision=0.0000 recall=0.0000
- 锡尖: gt=12 tp=10 fp=4 fn=2 precision=0.7143 recall=0.8333
- 锡膏: gt=21 tp=15 fp=10 fn=4 precision=0.6000 recall=0.7143
