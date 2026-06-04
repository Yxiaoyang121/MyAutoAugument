# Validation Error Diagnosis

- Status: completed
- TP: 694
- FP: 302
- FN: 196
- Precision: 0.6968
- Recall: 0.7669

## Diagnosis Vector
- small_object_score: 0.2937
- low_contrast_score: 0.5770
- class_imbalance_score: 0.9369
- localization_score: 0.0166
- false_positive_score: 0.3032

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=23 fn=0 precision=0.7356 recall=1.0000
- OK3: gt=274 tp=274 fp=70 fn=0 precision=0.7965 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=0 fn=1 precision=1.0000 recall=0.8750
- 开裂: gt=2 tp=2 fp=5 fn=0 precision=0.2857 recall=1.0000
- 油污: gt=18 tp=3 fp=26 fn=15 precision=0.1034 recall=0.1667
- 浅划伤: gt=13 tp=8 fp=5 fn=5 precision=0.6154 recall=0.6154
- 漏背锡: gt=46 tp=27 fp=19 fn=14 precision=0.5870 recall=0.5870
- 碰伤: gt=269 tp=164 fp=59 fn=103 precision=0.7354 recall=0.6097
- 脏污: gt=42 tp=32 fp=30 fn=8 precision=0.5161 recall=0.7619
- 轮廓划伤: gt=84 tp=46 fp=35 fn=33 precision=0.5679 recall=0.5476
- 锡丝残留: gt=12 tp=9 fp=5 fn=3 precision=0.6429 recall=0.7500
- 锡尖: gt=27 tp=24 fp=5 fn=3 precision=0.8276 recall=0.8889
- 锡膏: gt=46 tp=34 fp=20 fn=11 precision=0.6296 recall=0.7391
