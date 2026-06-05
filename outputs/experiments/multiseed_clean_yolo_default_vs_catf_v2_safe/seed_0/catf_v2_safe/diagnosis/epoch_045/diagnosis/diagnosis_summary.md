# Validation Error Diagnosis

- Status: completed
- TP: 724
- FP: 295
- FN: 170
- Precision: 0.7105
- Recall: 0.8000

## Diagnosis Vector
- small_object_score: 0.2632
- low_contrast_score: 0.5153
- class_imbalance_score: 0.8786
- localization_score: 0.0122
- false_positive_score: 0.2895

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=19 fn=0 precision=0.7711 recall=1.0000
- OK3: gt=274 tp=272 fp=64 fn=2 precision=0.8095 recall=0.9927
- 加强筋打伤: gt=8 tp=7 fp=2 fn=1 precision=0.7778 recall=0.8750
- 开裂: gt=2 tp=2 fp=8 fn=0 precision=0.2000 recall=1.0000
- 油污: gt=18 tp=6 fp=13 fn=12 precision=0.3158 recall=0.3333
- 浅划伤: gt=13 tp=6 fp=3 fn=7 precision=0.6667 recall=0.4615
- 漏背锡: gt=46 tp=33 fp=23 fn=11 precision=0.5893 recall=0.7174
- 碰伤: gt=269 tp=195 fp=94 fn=68 precision=0.6747 recall=0.7249
- 脏污: gt=42 tp=25 fp=25 fn=16 precision=0.5000 recall=0.5952
- 轮廓划伤: gt=84 tp=52 fp=24 fn=30 precision=0.6842 recall=0.6190
- 锡丝残留: gt=12 tp=11 fp=7 fn=1 precision=0.6111 recall=0.9167
- 锡尖: gt=27 tp=25 fp=2 fn=2 precision=0.9259 recall=0.9259
- 锡膏: gt=46 tp=26 fp=11 fn=20 precision=0.7027 recall=0.5652
