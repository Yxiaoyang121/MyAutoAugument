# Validation Error Diagnosis

- Status: completed
- TP: 729
- FP: 299
- FN: 160
- Precision: 0.7091
- Recall: 0.8055

## Diagnosis Vector
- small_object_score: 0.2309
- low_contrast_score: 0.5100
- class_imbalance_score: 0.9953
- localization_score: 0.0177
- false_positive_score: 0.2909

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=21 fn=0 precision=0.7529 recall=1.0000
- OK3: gt=274 tp=274 fp=68 fn=0 precision=0.8012 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=3 fn=1 precision=0.7000 recall=0.8750
- 开裂: gt=2 tp=0 fp=8 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=5 fp=6 fn=13 precision=0.4545 recall=0.2778
- 浅划伤: gt=13 tp=8 fp=7 fn=5 precision=0.5333 recall=0.6154
- 漏背锡: gt=46 tp=29 fp=29 fn=11 precision=0.5000 recall=0.6304
- 碰伤: gt=269 tp=200 fp=84 fn=64 precision=0.7042 recall=0.7435
- 脏污: gt=42 tp=23 fp=17 fn=18 precision=0.5750 recall=0.5476
- 轮廓划伤: gt=84 tp=50 fp=26 fn=30 precision=0.6579 recall=0.5952
- 锡丝残留: gt=12 tp=12 fp=8 fn=0 precision=0.6000 recall=1.0000
- 锡尖: gt=27 tp=25 fp=7 fn=2 precision=0.7812 recall=0.9259
- 锡膏: gt=46 tp=32 fp=15 fn=14 precision=0.6809 recall=0.6957
