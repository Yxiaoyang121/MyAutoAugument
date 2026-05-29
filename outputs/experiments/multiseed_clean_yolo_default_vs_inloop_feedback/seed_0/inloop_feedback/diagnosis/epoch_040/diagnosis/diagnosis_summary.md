# Validation Error Diagnosis

- Status: completed
- TP: 730
- FP: 313
- FN: 161
- Precision: 0.6999
- Recall: 0.8066

## Diagnosis Vector
- small_object_score: 0.2411
- low_contrast_score: 0.5354
- class_imbalance_score: 0.8397
- localization_score: 0.0155
- false_positive_score: 0.3001

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=21 fn=0 precision=0.7529 recall=1.0000
- OK3: gt=274 tp=274 fp=68 fn=0 precision=0.8012 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=0 fn=1 precision=1.0000 recall=0.8750
- 开裂: gt=2 tp=2 fp=11 fn=0 precision=0.1538 recall=1.0000
- 油污: gt=18 tp=8 fp=14 fn=10 precision=0.3636 recall=0.4444
- 浅划伤: gt=13 tp=9 fp=8 fn=3 precision=0.5294 recall=0.6923
- 漏背锡: gt=46 tp=29 fp=23 fn=14 precision=0.5577 recall=0.6304
- 碰伤: gt=269 tp=199 fp=82 fn=65 precision=0.7082 recall=0.7398
- 脏污: gt=42 tp=21 fp=25 fn=17 precision=0.4565 recall=0.5000
- 轮廓划伤: gt=84 tp=47 fp=36 fn=37 precision=0.5663 recall=0.5595
- 锡丝残留: gt=12 tp=9 fp=6 fn=3 precision=0.6000 recall=0.7500
- 锡尖: gt=27 tp=27 fp=5 fn=0 precision=0.8438 recall=1.0000
- 锡膏: gt=46 tp=34 fp=14 fn=11 precision=0.7083 recall=0.7391
