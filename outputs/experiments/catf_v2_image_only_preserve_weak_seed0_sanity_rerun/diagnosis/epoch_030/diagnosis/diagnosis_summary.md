# Validation Error Diagnosis

- Status: completed
- TP: 718
- FP: 351
- FN: 166
- Precision: 0.6717
- Recall: 0.7934

## Diagnosis Vector
- small_object_score: 0.2615
- low_contrast_score: 0.5560
- class_imbalance_score: 0.8397
- localization_score: 0.0232
- false_positive_score: 0.3283

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
- 开裂: gt=2 tp=2 fp=5 fn=0 precision=0.2857 recall=1.0000
- 油污: gt=18 tp=8 fp=58 fn=9 precision=0.1212 recall=0.4444
- 浅划伤: gt=13 tp=7 fp=0 fn=6 precision=1.0000 recall=0.5385
- 漏背锡: gt=46 tp=34 fp=41 fn=6 precision=0.4533 recall=0.7391
- 碰伤: gt=269 tp=176 fp=69 fn=86 precision=0.7184 recall=0.6543
- 脏污: gt=42 tp=24 fp=27 fn=14 precision=0.4706 recall=0.5714
- 轮廓划伤: gt=84 tp=56 fp=27 fn=28 precision=0.6747 recall=0.6667
- 锡丝残留: gt=12 tp=9 fp=7 fn=3 precision=0.5625 recall=0.7500
- 锡尖: gt=27 tp=26 fp=6 fn=1 precision=0.8125 recall=0.9630
- 锡膏: gt=46 tp=31 fp=22 fn=12 precision=0.5849 recall=0.6739
