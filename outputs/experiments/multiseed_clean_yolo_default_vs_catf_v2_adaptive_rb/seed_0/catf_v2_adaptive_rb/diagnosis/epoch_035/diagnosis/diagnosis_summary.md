# Validation Error Diagnosis

- Status: completed
- TP: 720
- FP: 335
- FN: 170
- Precision: 0.6825
- Recall: 0.7956

## Diagnosis Vector
- small_object_score: 0.2649
- low_contrast_score: 0.5424
- class_imbalance_score: 0.8591
- localization_score: 0.0166
- false_positive_score: 0.3175

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=21 fn=0 precision=0.7529 recall=1.0000
- OK3: gt=274 tp=274 fp=68 fn=0 precision=0.8012 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=1 fn=1 precision=0.8750 recall=0.8750
- 开裂: gt=2 tp=1 fp=3 fn=1 precision=0.2500 recall=0.5000
- 油污: gt=18 tp=7 fp=17 fn=10 precision=0.2917 recall=0.3889
- 浅划伤: gt=13 tp=7 fp=7 fn=6 precision=0.5000 recall=0.5385
- 漏背锡: gt=46 tp=35 fp=34 fn=11 precision=0.5072 recall=0.7609
- 碰伤: gt=269 tp=192 fp=97 fn=71 precision=0.6644 recall=0.7138
- 脏污: gt=42 tp=24 fp=24 fn=18 precision=0.5000 recall=0.5714
- 轮廓划伤: gt=84 tp=46 fp=32 fn=32 precision=0.5897 recall=0.5476
- 锡丝残留: gt=12 tp=9 fp=7 fn=3 precision=0.5625 recall=0.7500
- 锡尖: gt=27 tp=23 fp=7 fn=4 precision=0.7667 recall=0.8519
- 锡膏: gt=46 tp=31 fp=17 fn=13 precision=0.6458 recall=0.6739
