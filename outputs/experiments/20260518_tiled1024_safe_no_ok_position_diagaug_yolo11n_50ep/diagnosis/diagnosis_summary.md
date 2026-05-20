# Validation Error Diagnosis

- Status: completed
- TP: 668
- FP: 172
- FN: 215
- Precision: 0.7952
- Recall: 0.7381

## Diagnosis Vector
- small_object_score: 0.3294
- low_contrast_score: 0.5065
- class_imbalance_score: 0.9953
- localization_score: 0.0243
- false_positive_score: 0.2048

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=6 fn=0 precision=0.9143 recall=1.0000
- OK3: gt=274 tp=269 fp=8 fn=5 precision=0.9711 recall=0.9818
- 加强筋打伤: gt=8 tp=7 fp=0 fn=1 precision=1.0000 recall=0.8750
- 开裂: gt=2 tp=0 fp=2 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=5 fp=19 fn=13 precision=0.2083 recall=0.2778
- 浅划伤: gt=13 tp=7 fp=3 fn=6 precision=0.7000 recall=0.5385
- 漏背锡: gt=46 tp=32 fp=23 fn=11 precision=0.5818 recall=0.6957
- 碰伤: gt=269 tp=165 fp=83 fn=92 precision=0.6653 recall=0.6134
- 脏污: gt=42 tp=18 fp=9 fn=20 precision=0.6667 recall=0.4286
- 轮廓划伤: gt=84 tp=47 fp=9 fn=35 precision=0.8393 recall=0.5595
- 锡丝残留: gt=12 tp=5 fp=1 fn=7 precision=0.8333 recall=0.4167
- 锡尖: gt=27 tp=22 fp=4 fn=5 precision=0.8462 recall=0.8148
- 锡膏: gt=46 tp=27 fp=5 fn=18 precision=0.8438 recall=0.5870
