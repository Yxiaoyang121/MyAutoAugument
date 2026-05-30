# Validation Error Diagnosis

- Status: completed
- TP: 696
- FP: 300
- FN: 192
- Precision: 0.6988
- Recall: 0.7691

## Diagnosis Vector
- small_object_score: 0.2886
- low_contrast_score: 0.5404
- class_imbalance_score: 0.9953
- localization_score: 0.0188
- false_positive_score: 0.3012

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=26 fn=0 precision=0.7111 recall=1.0000
- OK3: gt=274 tp=273 fp=70 fn=1 precision=0.7959 recall=0.9964
- 加强筋打伤: gt=8 tp=7 fp=4 fn=0 precision=0.6364 recall=0.8750
- 开裂: gt=2 tp=0 fp=2 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=6 fp=35 fn=12 precision=0.1463 recall=0.3333
- 浅划伤: gt=13 tp=6 fp=1 fn=7 precision=0.8571 recall=0.4615
- 漏背锡: gt=46 tp=25 fp=22 fn=14 precision=0.5319 recall=0.5435
- 碰伤: gt=269 tp=172 fp=65 fn=92 precision=0.7257 recall=0.6394
- 脏污: gt=42 tp=23 fp=19 fn=18 precision=0.5476 recall=0.5476
- 轮廓划伤: gt=84 tp=51 fp=33 fn=31 precision=0.6071 recall=0.6071
- 锡丝残留: gt=12 tp=8 fp=5 fn=3 precision=0.6154 recall=0.6667
- 锡尖: gt=27 tp=21 fp=2 fn=6 precision=0.9130 recall=0.7778
- 锡膏: gt=46 tp=40 fp=16 fn=6 precision=0.7143 recall=0.8696
