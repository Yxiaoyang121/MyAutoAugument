# Validation Error Diagnosis

- Status: completed
- TP: 515
- FP: 228
- FN: 375
- Precision: 0.6931
- Recall: 0.5691

## Diagnosis Vector
- small_object_score: 0.5246
- low_contrast_score: 0.5000
- class_imbalance_score: 0.9953
- localization_score: 0.0166
- false_positive_score: 0.3069

## Issues
- small_object_low_recall severity=medium suggestions=tiling, object-aware-crop, scale, copy-paste, mild-geometry
- background_interference severity=medium suggestions=background-diversity, light-noise, illumination-jitter
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=42 fn=0 precision=0.6038 recall=1.0000
- OK3: gt=274 tp=264 fp=40 fn=10 precision=0.8684 recall=0.9635
- 加强筋打伤: gt=8 tp=0 fp=0 fn=8 precision=0.0000 recall=0.0000
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=0 fp=0 fn=18 precision=0.0000 recall=0.0000
- 浅划伤: gt=13 tp=6 fp=45 fn=7 precision=0.1176 recall=0.4615
- 漏背锡: gt=46 tp=23 fp=33 fn=20 precision=0.4107 recall=0.5000
- 碰伤: gt=269 tp=100 fp=34 fn=160 precision=0.7463 recall=0.3717
- 脏污: gt=42 tp=0 fp=0 fn=42 precision=0.0000 recall=0.0000
- 轮廓划伤: gt=84 tp=36 fp=31 fn=45 precision=0.5373 recall=0.4286
- 锡丝残留: gt=12 tp=0 fp=0 fn=12 precision=0.0000 recall=0.0000
- 锡尖: gt=27 tp=22 fp=3 fn=5 precision=0.8800 recall=0.8148
- 锡膏: gt=46 tp=0 fp=0 fn=46 precision=0.0000 recall=0.0000
