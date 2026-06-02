# Validation Error Diagnosis

- Status: completed
- TP: 674
- FP: 333
- FN: 212
- Precision: 0.6693
- Recall: 0.7448

## Diagnosis Vector
- small_object_score: 0.3158
- low_contrast_score: 0.5050
- class_imbalance_score: 0.9953
- localization_score: 0.0210
- false_positive_score: 0.3307

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=21 fn=0 precision=0.7529 recall=1.0000
- OK3: gt=274 tp=274 fp=66 fn=0 precision=0.8059 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=0 fn=1 precision=1.0000 recall=0.8750
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=3 fp=25 fn=15 precision=0.1071 recall=0.1667
- 浅划伤: gt=13 tp=6 fp=4 fn=7 precision=0.6000 recall=0.4615
- 漏背锡: gt=46 tp=26 fp=20 fn=16 precision=0.5652 recall=0.5652
- 碰伤: gt=269 tp=172 fp=92 fn=91 precision=0.6515 recall=0.6394
- 脏污: gt=42 tp=15 fp=23 fn=25 precision=0.3947 recall=0.3571
- 轮廓划伤: gt=84 tp=47 fp=45 fn=32 precision=0.5109 recall=0.5595
- 锡丝残留: gt=12 tp=7 fp=6 fn=5 precision=0.5385 recall=0.5833
- 锡尖: gt=27 tp=25 fp=19 fn=1 precision=0.5682 recall=0.9259
- 锡膏: gt=46 tp=28 fp=12 fn=17 precision=0.7000 recall=0.6087
