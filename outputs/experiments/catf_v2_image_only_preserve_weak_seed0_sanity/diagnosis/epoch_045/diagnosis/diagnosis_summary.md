# Validation Error Diagnosis

- Status: completed
- TP: 716
- FP: 286
- FN: 173
- Precision: 0.7146
- Recall: 0.7912

## Diagnosis Vector
- small_object_score: 0.2615
- low_contrast_score: 0.4960
- class_imbalance_score: 0.8980
- localization_score: 0.0177
- false_positive_score: 0.2854

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=20 fn=0 precision=0.7619 recall=1.0000
- OK3: gt=274 tp=270 fp=67 fn=4 precision=0.8012 recall=0.9854
- 加强筋打伤: gt=8 tp=3 fp=1 fn=5 precision=0.7500 recall=0.3750
- 开裂: gt=2 tp=1 fp=12 fn=1 precision=0.0769 recall=0.5000
- 油污: gt=18 tp=5 fp=7 fn=13 precision=0.4167 recall=0.2778
- 浅划伤: gt=13 tp=6 fp=3 fn=7 precision=0.6667 recall=0.4615
- 漏背锡: gt=46 tp=34 fp=20 fn=10 precision=0.6296 recall=0.7391
- 碰伤: gt=269 tp=192 fp=83 fn=69 precision=0.6982 recall=0.7138
- 脏污: gt=42 tp=25 fp=21 fn=17 precision=0.5435 recall=0.5952
- 轮廓划伤: gt=84 tp=57 fp=35 fn=23 precision=0.6196 recall=0.6786
- 锡丝残留: gt=12 tp=9 fp=5 fn=3 precision=0.6429 recall=0.7500
- 锡尖: gt=27 tp=24 fp=0 fn=3 precision=1.0000 recall=0.8889
- 锡膏: gt=46 tp=26 fp=12 fn=18 precision=0.6842 recall=0.5652
