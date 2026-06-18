# Validation Error Diagnosis

- Status: completed
- TP: 727
- FP: 292
- FN: 162
- Precision: 0.7134
- Recall: 0.8033

## Diagnosis Vector
- small_object_score: 0.2547
- low_contrast_score: 0.5673
- class_imbalance_score: 0.9175
- localization_score: 0.0177
- false_positive_score: 0.2866

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=20 fn=0 precision=0.7619 recall=1.0000
- OK3: gt=274 tp=273 fp=65 fn=1 precision=0.8077 recall=0.9964
- 加强筋打伤: gt=8 tp=6 fp=0 fn=1 precision=1.0000 recall=0.7500
- 开裂: gt=2 tp=2 fp=0 fn=0 precision=1.0000 recall=1.0000
- 油污: gt=18 tp=4 fp=21 fn=13 precision=0.1600 recall=0.2222
- 浅划伤: gt=13 tp=7 fp=4 fn=6 precision=0.6364 recall=0.5385
- 漏背锡: gt=46 tp=34 fp=17 fn=6 precision=0.6667 recall=0.7391
- 碰伤: gt=269 tp=196 fp=80 fn=68 precision=0.7101 recall=0.7286
- 脏污: gt=42 tp=24 fp=15 fn=18 precision=0.6154 recall=0.5714
- 轮廓划伤: gt=84 tp=49 fp=49 fn=32 precision=0.5000 recall=0.5833
- 锡丝残留: gt=12 tp=11 fp=7 fn=1 precision=0.6111 recall=0.9167
- 锡尖: gt=27 tp=27 fp=4 fn=0 precision=0.8710 recall=1.0000
- 锡膏: gt=46 tp=30 fp=10 fn=16 precision=0.7500 recall=0.6522
