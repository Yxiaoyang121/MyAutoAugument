# Validation Error Diagnosis

- Status: completed
- TP: 737
- FP: 310
- FN: 159
- Precision: 0.7039
- Recall: 0.8144

## Diagnosis Vector
- small_object_score: 0.2377
- low_contrast_score: 0.5035
- class_imbalance_score: 0.8786
- localization_score: 0.0099
- false_positive_score: 0.2961

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=21 fn=0 precision=0.7529 recall=1.0000
- OK3: gt=274 tp=274 fp=66 fn=0 precision=0.8059 recall=1.0000
- 加强筋打伤: gt=8 tp=4 fp=1 fn=4 precision=0.8000 recall=0.5000
- 开裂: gt=2 tp=1 fp=4 fn=1 precision=0.2000 recall=0.5000
- 油污: gt=18 tp=6 fp=23 fn=12 precision=0.2069 recall=0.3333
- 浅划伤: gt=13 tp=8 fp=6 fn=5 precision=0.5714 recall=0.6154
- 漏背锡: gt=46 tp=33 fp=18 fn=12 precision=0.6471 recall=0.7174
- 碰伤: gt=269 tp=207 fp=80 fn=57 precision=0.7213 recall=0.7695
- 脏污: gt=42 tp=24 fp=20 fn=16 precision=0.5455 recall=0.5714
- 轮廓划伤: gt=84 tp=54 fp=28 fn=29 precision=0.6585 recall=0.6429
- 锡丝残留: gt=12 tp=9 fp=6 fn=3 precision=0.6000 recall=0.7500
- 锡尖: gt=27 tp=21 fp=13 fn=6 precision=0.6176 recall=0.7778
- 锡膏: gt=46 tp=32 fp=24 fn=14 precision=0.5714 recall=0.6957
