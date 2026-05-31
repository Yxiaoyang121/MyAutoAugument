# Validation Error Diagnosis

- Status: completed
- TP: 718
- FP: 379
- FN: 167
- Precision: 0.6545
- Recall: 0.7934

## Diagnosis Vector
- small_object_score: 0.2547
- low_contrast_score: 0.5308
- class_imbalance_score: 0.9953
- localization_score: 0.0221
- false_positive_score: 0.3455

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=22 fn=0 precision=0.7442 recall=1.0000
- OK3: gt=274 tp=270 fp=67 fn=4 precision=0.8012 recall=0.9854
- 加强筋打伤: gt=8 tp=7 fp=2 fn=1 precision=0.7778 recall=0.8750
- 开裂: gt=2 tp=0 fp=1 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=4 fp=11 fn=14 precision=0.2667 recall=0.2222
- 浅划伤: gt=13 tp=4 fp=0 fn=9 precision=1.0000 recall=0.3077
- 漏背锡: gt=46 tp=36 fp=42 fn=7 precision=0.4615 recall=0.7826
- 碰伤: gt=269 tp=203 fp=108 fn=61 precision=0.6527 recall=0.7546
- 脏污: gt=42 tp=24 fp=30 fn=15 precision=0.4444 recall=0.5714
- 轮廓划伤: gt=84 tp=45 fp=60 fn=31 precision=0.4286 recall=0.5357
- 锡丝残留: gt=12 tp=1 fp=2 fn=11 precision=0.3333 recall=0.0833
- 锡尖: gt=27 tp=24 fp=5 fn=3 precision=0.8276 recall=0.8889
- 锡膏: gt=46 tp=36 fp=29 fn=9 precision=0.5538 recall=0.7826
