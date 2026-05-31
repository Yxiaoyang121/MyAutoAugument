# Validation Error Diagnosis

- Status: completed
- TP: 562
- FP: 185
- FN: 336
- Precision: 0.7523
- Recall: 0.6210

## Diagnosis Vector
- small_object_score: 0.4533
- low_contrast_score: 0.4598
- class_imbalance_score: 0.9953
- localization_score: 0.0077
- false_positive_score: 0.2477

## Issues
- small_object_low_recall severity=medium suggestions=tiling, object-aware-crop, scale, copy-paste, mild-geometry
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=18 fn=0 precision=0.7805 recall=1.0000
- OK3: gt=274 tp=266 fp=63 fn=8 precision=0.8085 recall=0.9708
- 加强筋打伤: gt=8 tp=0 fp=0 fn=8 precision=0.0000 recall=0.0000
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=4 fp=18 fn=14 precision=0.1818 recall=0.2222
- 浅划伤: gt=13 tp=5 fp=2 fn=8 precision=0.7143 recall=0.3846
- 漏背锡: gt=46 tp=13 fp=3 fn=31 precision=0.8125 recall=0.2826
- 碰伤: gt=269 tp=143 fp=56 fn=122 precision=0.7186 recall=0.5316
- 脏污: gt=42 tp=9 fp=5 fn=33 precision=0.6429 recall=0.2143
- 轮廓划伤: gt=84 tp=34 fp=17 fn=49 precision=0.6667 recall=0.4048
- 锡丝残留: gt=12 tp=0 fp=0 fn=12 precision=0.0000 recall=0.0000
- 锡尖: gt=27 tp=12 fp=0 fn=15 precision=1.0000 recall=0.4444
- 锡膏: gt=46 tp=12 fp=3 fn=34 precision=0.8000 recall=0.2609
