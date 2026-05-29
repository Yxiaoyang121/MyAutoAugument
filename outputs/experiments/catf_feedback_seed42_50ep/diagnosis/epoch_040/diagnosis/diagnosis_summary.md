# Validation Error Diagnosis

- Status: completed
- TP: 722
- FP: 298
- FN: 169
- Precision: 0.7078
- Recall: 0.7978

## Diagnosis Vector
- small_object_score: 0.2666
- low_contrast_score: 0.5293
- class_imbalance_score: 0.8786
- localization_score: 0.0155
- false_positive_score: 0.2922

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=24 fn=0 precision=0.7273 recall=1.0000
- OK3: gt=274 tp=272 fp=69 fn=2 precision=0.7977 recall=0.9927
- 加强筋打伤: gt=8 tp=7 fp=1 fn=1 precision=0.8750 recall=0.8750
- 开裂: gt=2 tp=2 fp=4 fn=0 precision=0.3333 recall=1.0000
- 油污: gt=18 tp=6 fp=19 fn=12 precision=0.2400 recall=0.3333
- 浅划伤: gt=13 tp=6 fp=0 fn=7 precision=1.0000 recall=0.4615
- 漏背锡: gt=46 tp=33 fp=30 fn=10 precision=0.5238 recall=0.7174
- 碰伤: gt=269 tp=198 fp=78 fn=67 precision=0.7174 recall=0.7361
- 脏污: gt=42 tp=19 fp=17 fn=20 precision=0.5278 recall=0.4524
- 轮廓划伤: gt=84 tp=45 fp=34 fn=35 precision=0.5696 recall=0.5357
- 锡丝残留: gt=12 tp=11 fp=6 fn=1 precision=0.6471 recall=0.9167
- 锡尖: gt=27 tp=27 fp=7 fn=0 precision=0.7941 recall=1.0000
- 锡膏: gt=46 tp=32 fp=9 fn=14 precision=0.7805 recall=0.6957
