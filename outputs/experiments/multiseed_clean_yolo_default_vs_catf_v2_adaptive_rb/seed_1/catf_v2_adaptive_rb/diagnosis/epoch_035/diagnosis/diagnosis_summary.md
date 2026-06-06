# Validation Error Diagnosis

- Status: completed
- TP: 710
- FP: 300
- FN: 181
- Precision: 0.7030
- Recall: 0.7845

## Diagnosis Vector
- small_object_score: 0.2767
- low_contrast_score: 0.5307
- class_imbalance_score: 0.8786
- localization_score: 0.0155
- false_positive_score: 0.2970

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=23 fn=0 precision=0.7356 recall=1.0000
- OK3: gt=274 tp=274 fp=69 fn=0 precision=0.7988 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=4 fn=1 precision=0.6364 recall=0.8750
- 开裂: gt=2 tp=2 fp=0 fn=0 precision=1.0000 recall=1.0000
- 油污: gt=18 tp=6 fp=16 fn=12 precision=0.2727 recall=0.3333
- 浅划伤: gt=13 tp=7 fp=4 fn=6 precision=0.6364 recall=0.5385
- 漏背锡: gt=46 tp=35 fp=43 fn=6 precision=0.4487 recall=0.7609
- 碰伤: gt=269 tp=181 fp=86 fn=84 precision=0.6779 recall=0.6729
- 脏污: gt=42 tp=21 fp=8 fn=21 precision=0.7241 recall=0.5000
- 轮廓划伤: gt=84 tp=50 fp=33 fn=29 precision=0.6024 recall=0.5952
- 锡丝残留: gt=12 tp=11 fp=8 fn=1 precision=0.5789 recall=0.9167
- 锡尖: gt=27 tp=21 fp=2 fn=6 precision=0.9130 recall=0.7778
- 锡膏: gt=46 tp=31 fp=4 fn=15 precision=0.8857 recall=0.6739
