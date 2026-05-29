# Validation Error Diagnosis

- Status: completed
- TP: 725
- FP: 286
- FN: 167
- Precision: 0.7171
- Recall: 0.8011

## Diagnosis Vector
- small_object_score: 0.2564
- low_contrast_score: 0.5719
- class_imbalance_score: 0.8786
- localization_score: 0.0144
- false_positive_score: 0.2829

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=20 fn=0 precision=0.7619 recall=1.0000
- OK3: gt=274 tp=273 fp=65 fn=1 precision=0.8077 recall=0.9964
- 加强筋打伤: gt=8 tp=7 fp=0 fn=1 precision=1.0000 recall=0.8750
- 开裂: gt=2 tp=2 fp=12 fn=0 precision=0.1429 recall=1.0000
- 油污: gt=18 tp=6 fp=17 fn=9 precision=0.2609 recall=0.3333
- 浅划伤: gt=13 tp=7 fp=3 fn=6 precision=0.7000 recall=0.5385
- 漏背锡: gt=46 tp=36 fp=20 fn=6 precision=0.6429 recall=0.7826
- 碰伤: gt=269 tp=187 fp=74 fn=78 precision=0.7165 recall=0.6952
- 脏污: gt=42 tp=24 fp=17 fn=17 precision=0.5854 recall=0.5714
- 轮廓划伤: gt=84 tp=51 fp=35 fn=32 precision=0.5930 recall=0.6071
- 锡丝残留: gt=12 tp=9 fp=6 fn=3 precision=0.6000 recall=0.7500
- 锡尖: gt=27 tp=27 fp=4 fn=0 precision=0.8710 recall=1.0000
- 锡膏: gt=46 tp=32 fp=13 fn=14 precision=0.7111 recall=0.6957
