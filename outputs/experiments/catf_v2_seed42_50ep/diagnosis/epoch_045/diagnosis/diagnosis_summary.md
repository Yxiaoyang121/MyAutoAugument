# Validation Error Diagnosis

- Status: completed
- TP: 703
- FP: 277
- FN: 185
- Precision: 0.7173
- Recall: 0.7768

## Diagnosis Vector
- small_object_score: 0.2835
- low_contrast_score: 0.5724
- class_imbalance_score: 0.8397
- localization_score: 0.0188
- false_positive_score: 0.2827

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=19 fn=0 precision=0.7711 recall=1.0000
- OK3: gt=274 tp=272 fp=65 fn=2 precision=0.8071 recall=0.9927
- 加强筋打伤: gt=8 tp=7 fp=0 fn=1 precision=1.0000 recall=0.8750
- 开裂: gt=2 tp=2 fp=1 fn=0 precision=0.6667 recall=1.0000
- 油污: gt=18 tp=8 fp=17 fn=10 precision=0.3200 recall=0.4444
- 浅划伤: gt=13 tp=6 fp=8 fn=7 precision=0.4286 recall=0.4615
- 漏背锡: gt=46 tp=32 fp=23 fn=10 precision=0.5818 recall=0.6957
- 碰伤: gt=269 tp=168 fp=69 fn=92 precision=0.7089 recall=0.6245
- 脏污: gt=42 tp=19 fp=16 fn=20 precision=0.5429 recall=0.4524
- 轮廓划伤: gt=84 tp=58 fp=37 fn=26 precision=0.6105 recall=0.6905
- 锡丝残留: gt=12 tp=10 fp=6 fn=2 precision=0.6250 recall=0.8333
- 锡尖: gt=27 tp=27 fp=5 fn=0 precision=0.8438 recall=1.0000
- 锡膏: gt=46 tp=30 fp=11 fn=15 precision=0.7317 recall=0.6522
