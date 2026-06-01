# Validation Error Diagnosis

- Status: completed
- TP: 592
- FP: 393
- FN: 280
- Precision: 0.6010
- Recall: 0.6541

## Diagnosis Vector
- small_object_score: 0.4177
- low_contrast_score: 0.4916
- class_imbalance_score: 0.9953
- localization_score: 0.0365
- false_positive_score: 0.3990

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=16 fn=0 precision=0.8000 recall=1.0000
- OK3: gt=274 tp=272 fp=64 fn=2 precision=0.8095 recall=0.9927
- 加强筋打伤: gt=8 tp=0 fp=0 fn=8 precision=0.0000 recall=0.0000
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=0 fp=7 fn=18 precision=0.0000 recall=0.0000
- 浅划伤: gt=13 tp=0 fp=0 fn=13 precision=0.0000 recall=0.0000
- 漏背锡: gt=46 tp=12 fp=35 fn=27 precision=0.2553 recall=0.2609
- 碰伤: gt=269 tp=157 fp=113 fn=99 precision=0.5815 recall=0.5836
- 脏污: gt=42 tp=16 fp=92 fn=26 precision=0.1481 recall=0.3810
- 轮廓划伤: gt=84 tp=25 fp=12 fn=51 precision=0.6757 recall=0.2976
- 锡丝残留: gt=12 tp=0 fp=0 fn=12 precision=0.0000 recall=0.0000
- 锡尖: gt=27 tp=24 fp=16 fn=3 precision=0.6000 recall=0.8889
- 锡膏: gt=46 tp=22 fp=38 fn=19 precision=0.3667 recall=0.4783
