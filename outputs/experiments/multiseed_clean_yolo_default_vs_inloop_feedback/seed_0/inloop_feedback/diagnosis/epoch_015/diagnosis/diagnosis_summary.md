# Validation Error Diagnosis

- Status: completed
- TP: 686
- FP: 417
- FN: 197
- Precision: 0.6219
- Recall: 0.7580

## Diagnosis Vector
- small_object_score: 0.2852
- low_contrast_score: 0.4718
- class_imbalance_score: 0.9953
- localization_score: 0.0243
- false_positive_score: 0.3781

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=21 fn=0 precision=0.7529 recall=1.0000
- OK3: gt=274 tp=274 fp=67 fn=0 precision=0.8035 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=0 fn=1 precision=1.0000 recall=0.8750
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=5 fp=26 fn=13 precision=0.1613 recall=0.2778
- 浅划伤: gt=13 tp=0 fp=0 fn=13 precision=0.0000 recall=0.0000
- 漏背锡: gt=46 tp=20 fp=25 fn=20 precision=0.4444 recall=0.4348
- 碰伤: gt=269 tp=190 fp=121 fn=71 precision=0.6109 recall=0.7063
- 脏污: gt=42 tp=20 fp=47 fn=20 precision=0.2985 recall=0.4762
- 轮廓划伤: gt=84 tp=51 fp=87 fn=29 precision=0.3696 recall=0.6071
- 锡丝残留: gt=12 tp=4 fp=7 fn=8 precision=0.3636 recall=0.3333
- 锡尖: gt=27 tp=23 fp=7 fn=4 precision=0.7667 recall=0.8519
- 锡膏: gt=46 tp=28 fp=9 fn=16 precision=0.7568 recall=0.6087
