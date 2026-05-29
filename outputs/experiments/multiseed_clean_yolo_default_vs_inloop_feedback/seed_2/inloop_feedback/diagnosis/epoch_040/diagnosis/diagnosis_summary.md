# Validation Error Diagnosis

- Status: completed
- TP: 701
- FP: 299
- FN: 182
- Precision: 0.7010
- Recall: 0.7746

## Diagnosis Vector
- small_object_score: 0.2835
- low_contrast_score: 0.5599
- class_imbalance_score: 0.8980
- localization_score: 0.0243
- false_positive_score: 0.2990

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=23 fn=0 precision=0.7356 recall=1.0000
- OK3: gt=274 tp=272 fp=70 fn=2 precision=0.7953 recall=0.9927
- 加强筋打伤: gt=8 tp=6 fp=6 fn=1 precision=0.5000 recall=0.7500
- 开裂: gt=2 tp=2 fp=4 fn=0 precision=0.3333 recall=1.0000
- 油污: gt=18 tp=5 fp=19 fn=13 precision=0.2083 recall=0.2778
- 浅划伤: gt=13 tp=7 fp=4 fn=5 precision=0.6364 recall=0.5385
- 漏背锡: gt=46 tp=29 fp=20 fn=12 precision=0.5918 recall=0.6304
- 碰伤: gt=269 tp=180 fp=72 fn=83 precision=0.7143 recall=0.6691
- 脏污: gt=42 tp=22 fp=24 fn=16 precision=0.4783 recall=0.5238
- 轮廓划伤: gt=84 tp=47 fp=27 fn=34 precision=0.6351 recall=0.5595
- 锡丝残留: gt=12 tp=9 fp=6 fn=3 precision=0.6000 recall=0.7500
- 锡尖: gt=27 tp=26 fp=7 fn=0 precision=0.7879 recall=0.9630
- 锡膏: gt=46 tp=32 fp=17 fn=13 precision=0.6531 recall=0.6957
