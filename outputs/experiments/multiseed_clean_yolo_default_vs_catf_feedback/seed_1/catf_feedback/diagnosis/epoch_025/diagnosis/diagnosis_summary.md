# Validation Error Diagnosis

- Status: completed
- TP: 669
- FP: 262
- FN: 221
- Precision: 0.7186
- Recall: 0.7392

## Diagnosis Vector
- small_object_score: 0.3362
- low_contrast_score: 0.5371
- class_imbalance_score: 0.9953
- localization_score: 0.0166
- false_positive_score: 0.2814

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=21 fn=0 precision=0.7529 recall=1.0000
- OK3: gt=274 tp=274 fp=66 fn=0 precision=0.8059 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=0 fn=1 precision=1.0000 recall=0.8750
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=4 fp=15 fn=14 precision=0.2105 recall=0.2222
- 浅划伤: gt=13 tp=6 fp=6 fn=7 precision=0.5000 recall=0.4615
- 漏背锡: gt=46 tp=37 fp=45 fn=6 precision=0.4512 recall=0.8043
- 碰伤: gt=269 tp=158 fp=53 fn=107 precision=0.7488 recall=0.5874
- 脏污: gt=42 tp=25 fp=14 fn=15 precision=0.6410 recall=0.5952
- 轮廓划伤: gt=84 tp=46 fp=20 fn=33 precision=0.6970 recall=0.5476
- 锡丝残留: gt=12 tp=1 fp=3 fn=11 precision=0.2500 recall=0.0833
- 锡尖: gt=27 tp=16 fp=1 fn=11 precision=0.9412 recall=0.5926
- 锡膏: gt=46 tp=31 fp=18 fn=14 precision=0.6327 recall=0.6739
