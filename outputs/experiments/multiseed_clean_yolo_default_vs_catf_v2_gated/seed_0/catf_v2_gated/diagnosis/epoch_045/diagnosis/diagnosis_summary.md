# Validation Error Diagnosis

- Status: completed
- TP: 718
- FP: 297
- FN: 171
- Precision: 0.7074
- Recall: 0.7934

## Diagnosis Vector
- small_object_score: 0.2615
- low_contrast_score: 0.5365
- class_imbalance_score: 0.9953
- localization_score: 0.0177
- false_positive_score: 0.2926

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=23 fn=0 precision=0.7356 recall=1.0000
- OK3: gt=274 tp=272 fp=61 fn=2 precision=0.8168 recall=0.9927
- 加强筋打伤: gt=8 tp=6 fp=0 fn=2 precision=1.0000 recall=0.7500
- 开裂: gt=2 tp=0 fp=14 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=7 fp=13 fn=11 precision=0.3500 recall=0.3889
- 浅划伤: gt=13 tp=6 fp=0 fn=7 precision=1.0000 recall=0.4615
- 漏背锡: gt=46 tp=35 fp=27 fn=6 precision=0.5645 recall=0.7609
- 碰伤: gt=269 tp=188 fp=81 fn=73 precision=0.6989 recall=0.6989
- 脏污: gt=42 tp=28 fp=27 fn=14 precision=0.5091 recall=0.6667
- 轮廓划伤: gt=84 tp=51 fp=25 fn=31 precision=0.6711 recall=0.6071
- 锡丝残留: gt=12 tp=9 fp=5 fn=3 precision=0.6429 recall=0.7500
- 锡尖: gt=27 tp=25 fp=6 fn=2 precision=0.8065 recall=0.9259
- 锡膏: gt=46 tp=27 fp=15 fn=18 precision=0.6429 recall=0.5870
