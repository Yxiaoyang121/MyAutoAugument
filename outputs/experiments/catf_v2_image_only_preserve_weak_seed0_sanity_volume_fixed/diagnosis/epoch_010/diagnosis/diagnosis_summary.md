# Validation Error Diagnosis

- Status: completed
- TP: 639
- FP: 423
- FN: 242
- Precision: 0.6017
- Recall: 0.7061

## Diagnosis Vector
- small_object_score: 0.3599
- low_contrast_score: 0.4876
- class_imbalance_score: 0.9953
- localization_score: 0.0265
- false_positive_score: 0.3983

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=27 fn=0 precision=0.7033 recall=1.0000
- OK3: gt=274 tp=274 fp=66 fn=0 precision=0.8059 recall=1.0000
- 加强筋打伤: gt=8 tp=2 fp=0 fn=6 precision=1.0000 recall=0.2500
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=3 fp=85 fn=15 precision=0.0341 recall=0.1667
- 浅划伤: gt=13 tp=8 fp=11 fn=5 precision=0.4211 recall=0.6154
- 漏背锡: gt=46 tp=25 fp=19 fn=16 precision=0.5682 recall=0.5435
- 碰伤: gt=269 tp=179 fp=107 fn=81 precision=0.6259 recall=0.6654
- 脏污: gt=42 tp=8 fp=41 fn=30 precision=0.1633 recall=0.1905
- 轮廓划伤: gt=84 tp=28 fp=16 fn=51 precision=0.6364 recall=0.3333
- 锡丝残留: gt=12 tp=0 fp=0 fn=12 precision=0.0000 recall=0.0000
- 锡尖: gt=27 tp=11 fp=3 fn=16 precision=0.7857 recall=0.4074
- 锡膏: gt=46 tp=37 fp=48 fn=8 precision=0.4353 recall=0.8043
