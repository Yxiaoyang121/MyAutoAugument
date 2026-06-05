# Validation Error Diagnosis

- Status: completed
- TP: 732
- FP: 543
- FN: 150
- Precision: 0.5741
- Recall: 0.8088

## Diagnosis Vector
- small_object_score: 0.2224
- low_contrast_score: 0.5133
- class_imbalance_score: 0.9953
- localization_score: 0.0254
- false_positive_score: 0.4259

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=31 fn=0 precision=0.6737 recall=1.0000
- OK3: gt=274 tp=273 fp=68 fn=0 precision=0.8006 recall=0.9964
- 加强筋打伤: gt=8 tp=7 fp=9 fn=1 precision=0.4375 recall=0.8750
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=1 fp=15 fn=16 precision=0.0625 recall=0.0556
- 浅划伤: gt=13 tp=8 fp=28 fn=5 precision=0.2222 recall=0.6154
- 漏背锡: gt=46 tp=30 fp=37 fn=12 precision=0.4478 recall=0.6522
- 碰伤: gt=269 tp=204 fp=129 fn=57 precision=0.6126 recall=0.7584
- 脏污: gt=42 tp=24 fp=49 fn=14 precision=0.3288 recall=0.5714
- 轮廓划伤: gt=84 tp=55 fp=103 fn=26 precision=0.3481 recall=0.6548
- 锡丝残留: gt=12 tp=8 fp=7 fn=4 precision=0.5333 recall=0.6667
- 锡尖: gt=27 tp=25 fp=11 fn=2 precision=0.6944 recall=0.9259
- 锡膏: gt=46 tp=33 fp=56 fn=11 precision=0.3708 recall=0.7174
