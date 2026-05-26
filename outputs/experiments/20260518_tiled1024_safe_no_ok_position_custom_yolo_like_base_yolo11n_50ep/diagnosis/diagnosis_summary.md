# Validation Error Diagnosis

- Status: completed
- TP: 722
- FP: 319
- FN: 170
- Precision: 0.6936
- Recall: 0.7978

## Diagnosis Vector
- small_object_score: 0.2699
- low_contrast_score: 0.5600
- class_imbalance_score: 0.8397
- localization_score: 0.0144
- false_positive_score: 0.3064

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=15 fn=0 precision=0.8101 recall=1.0000
- OK3: gt=274 tp=273 fp=59 fn=1 precision=0.8223 recall=0.9964
- 加强筋打伤: gt=8 tp=4 fp=1 fn=4 precision=0.8000 recall=0.5000
- 开裂: gt=2 tp=2 fp=8 fn=0 precision=0.2000 recall=1.0000
- 油污: gt=18 tp=8 fp=30 fn=10 precision=0.2105 recall=0.4444
- 浅划伤: gt=13 tp=8 fp=4 fn=5 precision=0.6667 recall=0.6154
- 漏背锡: gt=46 tp=34 fp=12 fn=9 precision=0.7391 recall=0.7391
- 碰伤: gt=269 tp=191 fp=107 fn=71 precision=0.6409 recall=0.7100
- 脏污: gt=42 tp=24 fp=22 fn=18 precision=0.5217 recall=0.5714
- 轮廓划伤: gt=84 tp=45 fp=23 fn=36 precision=0.6618 recall=0.5357
- 锡丝残留: gt=12 tp=9 fp=6 fn=3 precision=0.6000 recall=0.7500
- 锡尖: gt=27 tp=27 fp=7 fn=0 precision=0.7941 recall=1.0000
- 锡膏: gt=46 tp=33 fp=25 fn=13 precision=0.5690 recall=0.7174
