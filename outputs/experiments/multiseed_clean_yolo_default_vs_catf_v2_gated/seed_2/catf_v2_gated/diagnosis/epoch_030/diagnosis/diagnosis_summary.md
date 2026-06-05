# Validation Error Diagnosis

- Status: completed
- TP: 692
- FP: 279
- FN: 192
- Precision: 0.7127
- Recall: 0.7646

## Diagnosis Vector
- small_object_score: 0.2937
- low_contrast_score: 0.5359
- class_imbalance_score: 0.9369
- localization_score: 0.0232
- false_positive_score: 0.2873

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=25 fn=0 precision=0.7191 recall=1.0000
- OK3: gt=274 tp=273 fp=69 fn=1 precision=0.7982 recall=0.9964
- 加强筋打伤: gt=8 tp=7 fp=0 fn=1 precision=1.0000 recall=0.8750
- 开裂: gt=2 tp=2 fp=2 fn=0 precision=0.5000 recall=1.0000
- 油污: gt=18 tp=3 fp=19 fn=12 precision=0.1364 recall=0.1667
- 浅划伤: gt=13 tp=6 fp=5 fn=7 precision=0.5455 recall=0.4615
- 漏背锡: gt=46 tp=28 fp=21 fn=12 precision=0.5714 recall=0.6087
- 碰伤: gt=269 tp=168 fp=57 fn=97 precision=0.7467 recall=0.6245
- 脏污: gt=42 tp=27 fp=22 fn=12 precision=0.5510 recall=0.6429
- 轮廓划伤: gt=84 tp=53 fp=31 fn=26 precision=0.6310 recall=0.6310
- 锡丝残留: gt=12 tp=9 fp=5 fn=3 precision=0.6429 recall=0.7500
- 锡尖: gt=27 tp=22 fp=6 fn=5 precision=0.7857 recall=0.8148
- 锡膏: gt=46 tp=30 fp=17 fn=16 precision=0.6383 recall=0.6522
