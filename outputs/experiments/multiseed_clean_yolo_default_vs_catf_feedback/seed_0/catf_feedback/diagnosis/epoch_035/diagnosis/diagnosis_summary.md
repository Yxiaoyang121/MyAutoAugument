# Validation Error Diagnosis

- Status: completed
- TP: 716
- FP: 290
- FN: 173
- Precision: 0.7117
- Recall: 0.7912

## Diagnosis Vector
- small_object_score: 0.2699
- low_contrast_score: 0.5214
- class_imbalance_score: 0.8397
- localization_score: 0.0177
- false_positive_score: 0.2883

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=21 fn=0 precision=0.7529 recall=1.0000
- OK3: gt=274 tp=274 fp=67 fn=0 precision=0.8035 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=1 fn=1 precision=0.8750 recall=0.8750
- 开裂: gt=2 tp=2 fp=9 fn=0 precision=0.1818 recall=1.0000
- 油污: gt=18 tp=8 fp=28 fn=8 precision=0.2222 recall=0.4444
- 浅划伤: gt=13 tp=9 fp=6 fn=4 precision=0.6000 recall=0.6923
- 漏背锡: gt=46 tp=28 fp=9 fn=14 precision=0.7568 recall=0.6087
- 碰伤: gt=269 tp=180 fp=60 fn=84 precision=0.7500 recall=0.6691
- 脏污: gt=42 tp=22 fp=21 fn=20 precision=0.5116 recall=0.5238
- 轮廓划伤: gt=84 tp=59 fp=39 fn=20 precision=0.6020 recall=0.7024
- 锡丝残留: gt=12 tp=6 fp=6 fn=6 precision=0.5000 recall=0.5000
- 锡尖: gt=27 tp=25 fp=7 fn=2 precision=0.7812 recall=0.9259
- 锡膏: gt=46 tp=32 fp=16 fn=14 precision=0.6667 recall=0.6957
