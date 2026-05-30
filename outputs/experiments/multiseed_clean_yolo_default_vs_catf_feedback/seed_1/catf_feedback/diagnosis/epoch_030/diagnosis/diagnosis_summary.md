# Validation Error Diagnosis

- Status: completed
- TP: 682
- FP: 345
- FN: 200
- Precision: 0.6641
- Recall: 0.7536

## Diagnosis Vector
- small_object_score: 0.3243
- low_contrast_score: 0.6050
- class_imbalance_score: 0.9175
- localization_score: 0.0254
- false_positive_score: 0.3359

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=24 fn=0 precision=0.7273 recall=1.0000
- OK3: gt=274 tp=274 fp=73 fn=0 precision=0.7896 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=5 fn=1 precision=0.5833 recall=0.8750
- 开裂: gt=2 tp=2 fp=0 fn=0 precision=1.0000 recall=1.0000
- 油污: gt=18 tp=4 fp=25 fn=14 precision=0.1379 recall=0.2222
- 浅划伤: gt=13 tp=6 fp=17 fn=7 precision=0.2609 recall=0.4615
- 漏背锡: gt=46 tp=31 fp=35 fn=8 precision=0.4697 recall=0.6739
- 碰伤: gt=269 tp=158 fp=62 fn=108 precision=0.7182 recall=0.5874
- 脏污: gt=42 tp=28 fp=28 fn=10 precision=0.5000 recall=0.6667
- 轮廓划伤: gt=84 tp=41 fp=40 fn=35 precision=0.5062 recall=0.4881
- 锡丝残留: gt=12 tp=8 fp=17 fn=4 precision=0.3200 recall=0.6667
- 锡尖: gt=27 tp=25 fp=6 fn=2 precision=0.8065 recall=0.9259
- 锡膏: gt=46 tp=34 fp=13 fn=11 precision=0.7234 recall=0.7391
