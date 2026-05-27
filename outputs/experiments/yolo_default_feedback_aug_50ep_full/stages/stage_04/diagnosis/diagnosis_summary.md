# Validation Error Diagnosis

- Status: completed
- TP: 708
- FP: 329
- FN: 179
- Precision: 0.6827
- Recall: 0.7823

## Diagnosis Vector
- small_object_score: 0.2683
- low_contrast_score: 0.5165
- class_imbalance_score: 0.9953
- localization_score: 0.0199
- false_positive_score: 0.3173

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=23 fn=0 precision=0.7356 recall=1.0000
- OK3: gt=274 tp=273 fp=64 fn=1 precision=0.8101 recall=0.9964
- 加强筋打伤: gt=8 tp=7 fp=0 fn=1 precision=1.0000 recall=0.8750
- 开裂: gt=2 tp=0 fp=14 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=6 fp=40 fn=12 precision=0.1304 recall=0.3333
- 浅划伤: gt=13 tp=6 fp=5 fn=7 precision=0.5455 recall=0.4615
- 漏背锡: gt=46 tp=30 fp=19 fn=11 precision=0.6122 recall=0.6522
- 碰伤: gt=269 tp=186 fp=72 fn=78 precision=0.7209 recall=0.6914
- 脏污: gt=42 tp=20 fp=21 fn=20 precision=0.4878 recall=0.4762
- 轮廓划伤: gt=84 tp=52 fp=33 fn=27 precision=0.6118 recall=0.6190
- 锡丝残留: gt=12 tp=8 fp=6 fn=4 precision=0.5714 recall=0.6667
- 锡尖: gt=27 tp=23 fp=2 fn=4 precision=0.9200 recall=0.8519
- 锡膏: gt=46 tp=33 fp=30 fn=12 precision=0.5238 recall=0.7174
