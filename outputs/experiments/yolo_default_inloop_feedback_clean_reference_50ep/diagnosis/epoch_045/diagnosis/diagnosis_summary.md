# Validation Error Diagnosis

- Status: completed
- TP: 737
- FP: 298
- FN: 146
- Precision: 0.7121
- Recall: 0.8144

## Diagnosis Vector
- small_object_score: 0.2377
- low_contrast_score: 0.5291
- class_imbalance_score: 0.8337
- localization_score: 0.0243
- false_positive_score: 0.2879

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=17 fn=0 precision=0.7901 recall=1.0000
- OK3: gt=274 tp=273 fp=65 fn=1 precision=0.8077 recall=0.9964
- 加强筋打伤: gt=8 tp=7 fp=0 fn=1 precision=1.0000 recall=0.8750
- 开裂: gt=2 tp=2 fp=4 fn=0 precision=0.3333 recall=1.0000
- 油污: gt=18 tp=10 fp=19 fn=7 precision=0.3448 recall=0.5556
- 浅划伤: gt=13 tp=6 fp=4 fn=7 precision=0.6000 recall=0.4615
- 漏背锡: gt=46 tp=32 fp=18 fn=11 precision=0.6400 recall=0.6957
- 碰伤: gt=269 tp=192 fp=89 fn=68 precision=0.6833 recall=0.7138
- 脏污: gt=42 tp=27 fp=17 fn=13 precision=0.6136 recall=0.6429
- 轮廓划伤: gt=84 tp=50 fp=38 fn=28 precision=0.5682 recall=0.5952
- 锡丝残留: gt=12 tp=12 fp=6 fn=0 precision=0.6667 recall=1.0000
- 锡尖: gt=27 tp=27 fp=7 fn=0 precision=0.7941 recall=1.0000
- 锡膏: gt=46 tp=35 fp=14 fn=10 precision=0.7143 recall=0.7609
