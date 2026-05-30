# Validation Error Diagnosis

- Status: completed
- TP: 723
- FP: 273
- FN: 165
- Precision: 0.7259
- Recall: 0.7989

## Diagnosis Vector
- small_object_score: 0.2666
- low_contrast_score: 0.5333
- class_imbalance_score: 0.8397
- localization_score: 0.0188
- false_positive_score: 0.2741

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=20 fn=0 precision=0.7619 recall=1.0000
- OK3: gt=274 tp=274 fp=61 fn=0 precision=0.8179 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=0 fn=1 precision=1.0000 recall=0.8750
- 开裂: gt=2 tp=2 fp=9 fn=0 precision=0.1818 recall=1.0000
- 油污: gt=18 tp=8 fp=20 fn=10 precision=0.2857 recall=0.4444
- 浅划伤: gt=13 tp=9 fp=1 fn=4 precision=0.9000 recall=0.6923
- 漏背锡: gt=46 tp=34 fp=17 fn=8 precision=0.6667 recall=0.7391
- 碰伤: gt=269 tp=186 fp=74 fn=79 precision=0.7154 recall=0.6914
- 脏污: gt=42 tp=23 fp=15 fn=17 precision=0.6053 recall=0.5476
- 轮廓划伤: gt=84 tp=55 fp=40 fn=23 precision=0.5789 recall=0.6548
- 锡丝残留: gt=12 tp=9 fp=4 fn=2 precision=0.6923 recall=0.7500
- 锡尖: gt=27 tp=25 fp=1 fn=2 precision=0.9615 recall=0.9259
- 锡膏: gt=46 tp=27 fp=11 fn=19 precision=0.7105 recall=0.5870
