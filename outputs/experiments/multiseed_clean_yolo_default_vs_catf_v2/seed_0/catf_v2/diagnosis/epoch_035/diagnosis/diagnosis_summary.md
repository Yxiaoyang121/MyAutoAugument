# Validation Error Diagnosis

- Status: completed
- TP: 717
- FP: 294
- FN: 177
- Precision: 0.7092
- Recall: 0.7923

## Diagnosis Vector
- small_object_score: 0.2784
- low_contrast_score: 0.5621
- class_imbalance_score: 0.8397
- localization_score: 0.0122
- false_positive_score: 0.2908

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=25 fn=0 precision=0.7191 recall=1.0000
- OK3: gt=274 tp=274 fp=70 fn=0 precision=0.7965 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=0 fn=1 precision=1.0000 recall=0.8750
- 开裂: gt=2 tp=2 fp=7 fn=0 precision=0.2222 recall=1.0000
- 油污: gt=18 tp=8 fp=21 fn=10 precision=0.2759 recall=0.4444
- 浅划伤: gt=13 tp=7 fp=4 fn=6 precision=0.6364 recall=0.5385
- 漏背锡: gt=46 tp=36 fp=26 fn=9 precision=0.5806 recall=0.7826
- 碰伤: gt=269 tp=180 fp=64 fn=82 precision=0.7377 recall=0.6691
- 脏污: gt=42 tp=24 fp=30 fn=16 precision=0.4444 recall=0.5714
- 轮廓划伤: gt=84 tp=50 fp=23 fn=34 precision=0.6849 recall=0.5952
- 锡丝残留: gt=12 tp=8 fp=6 fn=4 precision=0.5714 recall=0.6667
- 锡尖: gt=27 tp=26 fp=8 fn=1 precision=0.7647 recall=0.9630
- 锡膏: gt=46 tp=31 fp=10 fn=14 precision=0.7561 recall=0.6739
