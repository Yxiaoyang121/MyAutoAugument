# Validation Error Diagnosis

- Status: completed
- TP: 719
- FP: 287
- FN: 162
- Precision: 0.7147
- Recall: 0.7945

## Diagnosis Vector
- small_object_score: 0.2699
- low_contrast_score: 0.5824
- class_imbalance_score: 0.8203
- localization_score: 0.0265
- false_positive_score: 0.2853

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=18 fn=0 precision=0.7805 recall=1.0000
- OK3: gt=274 tp=273 fp=66 fn=1 precision=0.8053 recall=0.9964
- 加强筋打伤: gt=8 tp=7 fp=0 fn=1 precision=1.0000 recall=0.8750
- 开裂: gt=2 tp=2 fp=1 fn=0 precision=0.6667 recall=1.0000
- 油污: gt=18 tp=9 fp=31 fn=9 precision=0.2250 recall=0.5000
- 浅划伤: gt=13 tp=7 fp=4 fn=5 precision=0.6364 recall=0.5385
- 漏背锡: gt=46 tp=32 fp=24 fn=8 precision=0.5714 recall=0.6957
- 碰伤: gt=269 tp=178 fp=78 fn=79 precision=0.6953 recall=0.6617
- 脏污: gt=42 tp=26 fp=16 fn=15 precision=0.6190 recall=0.6190
- 轮廓划伤: gt=84 tp=50 fp=27 fn=30 precision=0.6494 recall=0.5952
- 锡丝残留: gt=12 tp=11 fp=7 fn=1 precision=0.6111 recall=0.9167
- 锡尖: gt=27 tp=27 fp=2 fn=0 precision=0.9310 recall=1.0000
- 锡膏: gt=46 tp=33 fp=13 fn=13 precision=0.7174 recall=0.7174
