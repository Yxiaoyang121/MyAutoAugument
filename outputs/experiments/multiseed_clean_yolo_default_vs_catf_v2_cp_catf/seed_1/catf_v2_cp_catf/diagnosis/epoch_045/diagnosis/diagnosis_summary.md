# Validation Error Diagnosis

- Status: completed
- TP: 725
- FP: 289
- FN: 162
- Precision: 0.7150
- Recall: 0.8011

## Diagnosis Vector
- small_object_score: 0.2547
- low_contrast_score: 0.5531
- class_imbalance_score: 0.9175
- localization_score: 0.0199
- false_positive_score: 0.2850

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=20 fn=0 precision=0.7619 recall=1.0000
- OK3: gt=274 tp=273 fp=65 fn=1 precision=0.8077 recall=0.9964
- 加强筋打伤: gt=8 tp=6 fp=1 fn=1 precision=0.8571 recall=0.7500
- 开裂: gt=2 tp=1 fp=0 fn=1 precision=1.0000 recall=0.5000
- 油污: gt=18 tp=4 fp=22 fn=12 precision=0.1538 recall=0.2222
- 浅划伤: gt=13 tp=7 fp=4 fn=6 precision=0.6364 recall=0.5385
- 漏背锡: gt=46 tp=35 fp=23 fn=6 precision=0.6034 recall=0.7609
- 碰伤: gt=269 tp=195 fp=75 fn=68 precision=0.7222 recall=0.7249
- 脏污: gt=42 tp=25 fp=19 fn=17 precision=0.5682 recall=0.5952
- 轮廓划伤: gt=84 tp=51 fp=36 fn=29 precision=0.5862 recall=0.6071
- 锡丝残留: gt=12 tp=10 fp=8 fn=2 precision=0.5556 recall=0.8333
- 锡尖: gt=27 tp=27 fp=4 fn=0 precision=0.8710 recall=1.0000
- 锡膏: gt=46 tp=27 fp=12 fn=19 precision=0.6923 recall=0.5870
