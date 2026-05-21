# Validation Error Diagnosis

- Status: completed
- TP: 681
- FP: 150
- FN: 215
- Precision: 0.8195
- Recall: 0.7525

## Diagnosis Vector
- small_object_score: 0.3175
- low_contrast_score: 0.5644
- class_imbalance_score: 0.9175
- localization_score: 0.0099
- false_positive_score: 0.1805

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=2 fn=0 precision=0.9697 recall=1.0000
- OK3: gt=274 tp=270 fp=5 fn=4 precision=0.9818 recall=0.9854
- 加强筋打伤: gt=8 tp=6 fp=1 fn=2 precision=0.8571 recall=0.7500
- 开裂: gt=2 tp=2 fp=1 fn=0 precision=0.6667 recall=1.0000
- 油污: gt=18 tp=4 fp=13 fn=14 precision=0.2353 recall=0.2222
- 浅划伤: gt=13 tp=9 fp=11 fn=4 precision=0.4500 recall=0.6923
- 漏背锡: gt=46 tp=31 fp=22 fn=11 precision=0.5849 recall=0.6739
- 碰伤: gt=269 tp=157 fp=51 fn=108 precision=0.7548 recall=0.5836
- 脏污: gt=42 tp=16 fp=10 fn=26 precision=0.6154 recall=0.3810
- 轮廓划伤: gt=84 tp=57 fp=15 fn=26 precision=0.7917 recall=0.6786
- 锡丝残留: gt=12 tp=8 fp=5 fn=4 precision=0.6154 recall=0.6667
- 锡尖: gt=27 tp=23 fp=0 fn=4 precision=1.0000 recall=0.8519
- 锡膏: gt=46 tp=34 fp=14 fn=12 precision=0.7083 recall=0.7391
