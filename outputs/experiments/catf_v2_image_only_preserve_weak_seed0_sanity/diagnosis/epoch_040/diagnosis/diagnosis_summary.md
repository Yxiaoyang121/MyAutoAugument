# Validation Error Diagnosis

- Status: completed
- TP: 714
- FP: 300
- FN: 171
- Precision: 0.7041
- Recall: 0.7890

## Diagnosis Vector
- small_object_score: 0.2564
- low_contrast_score: 0.5219
- class_imbalance_score: 0.8980
- localization_score: 0.0221
- false_positive_score: 0.2959

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=21 fn=0 precision=0.7529 recall=1.0000
- OK3: gt=274 tp=273 fp=69 fn=1 precision=0.7982 recall=0.9964
- 加强筋打伤: gt=8 tp=7 fp=2 fn=1 precision=0.7778 recall=0.8750
- 开裂: gt=2 tp=2 fp=4 fn=0 precision=0.3333 recall=1.0000
- 油污: gt=18 tp=5 fp=7 fn=13 precision=0.4167 recall=0.2778
- 浅划伤: gt=13 tp=10 fp=12 fn=3 precision=0.4545 recall=0.7692
- 漏背锡: gt=46 tp=29 fp=27 fn=12 precision=0.5179 recall=0.6304
- 碰伤: gt=269 tp=187 fp=61 fn=77 precision=0.7540 recall=0.6952
- 脏污: gt=42 tp=22 fp=25 fn=17 precision=0.4681 recall=0.5238
- 轮廓划伤: gt=84 tp=51 fp=31 fn=27 precision=0.6220 recall=0.6071
- 锡丝残留: gt=12 tp=9 fp=6 fn=3 precision=0.6000 recall=0.7500
- 锡尖: gt=27 tp=27 fp=11 fn=0 precision=0.7105 recall=1.0000
- 锡膏: gt=46 tp=28 fp=24 fn=17 precision=0.5385 recall=0.6087
