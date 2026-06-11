# Validation Error Diagnosis

- Status: completed
- TP: 364
- FP: 113
- FN: 74
- Precision: 0.7631
- Recall: 0.8235

## Diagnosis Vector
- small_object_score: 0.2140
- low_contrast_score: 0.5534
- class_imbalance_score: 0.9236
- localization_score: 0.0090
- false_positive_score: 0.2369

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=38 tp=37 fp=2 fn=1 precision=0.9487 recall=0.9737
- OK3: gt=102 tp=102 fp=22 fn=0 precision=0.8226 recall=1.0000
- 加强筋打伤: gt=6 tp=6 fp=1 fn=0 precision=0.8571 recall=1.0000
- 开裂: gt=3 tp=3 fp=8 fn=0 precision=0.2727 recall=1.0000
- 油污: gt=17 tp=3 fp=2 fn=13 precision=0.6000 recall=0.1765
- 浅划伤: gt=14 tp=3 fp=7 fn=11 precision=0.3000 recall=0.2143
- 漏背锡: gt=32 tp=26 fp=11 fn=6 precision=0.7027 recall=0.8125
- 碰伤: gt=133 tp=106 fp=19 fn=27 precision=0.8480 recall=0.7970
- 脏污: gt=28 tp=14 fp=27 fn=12 precision=0.3415 recall=0.5000
- 轮廓划伤: gt=30 tp=29 fp=6 fn=0 precision=0.8286 recall=0.9667
- 锡丝残留: gt=6 tp=4 fp=2 fn=2 precision=0.6667 recall=0.6667
- 锡尖: gt=12 tp=11 fp=4 fn=1 precision=0.7333 recall=0.9167
- 锡膏: gt=21 tp=20 fp=2 fn=1 precision=0.9091 recall=0.9524
