# Validation Error Diagnosis

- Status: completed
- TP: 668
- FP: 357
- FN: 209
- Precision: 0.6517
- Recall: 0.7381

## Diagnosis Vector
- small_object_score: 0.3192
- low_contrast_score: 0.4821
- class_imbalance_score: 0.9953
- localization_score: 0.0309
- false_positive_score: 0.3483

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=25 fn=0 precision=0.7191 recall=1.0000
- OK3: gt=274 tp=274 fp=65 fn=0 precision=0.8083 recall=1.0000
- 加强筋打伤: gt=8 tp=6 fp=2 fn=2 precision=0.7500 recall=0.7500
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=4 fp=17 fn=14 precision=0.1905 recall=0.2222
- 浅划伤: gt=13 tp=4 fp=3 fn=9 precision=0.5714 recall=0.3077
- 漏背锡: gt=46 tp=26 fp=39 fn=14 precision=0.4000 recall=0.5652
- 碰伤: gt=269 tp=178 fp=115 fn=79 precision=0.6075 recall=0.6617
- 脏污: gt=42 tp=11 fp=15 fn=28 precision=0.4231 recall=0.2619
- 轮廓划伤: gt=84 tp=45 fp=30 fn=34 precision=0.6000 recall=0.5357
- 锡丝残留: gt=12 tp=0 fp=0 fn=12 precision=0.0000 recall=0.0000
- 锡尖: gt=27 tp=21 fp=15 fn=6 precision=0.5833 recall=0.7778
- 锡膏: gt=46 tp=35 fp=31 fn=9 precision=0.5303 recall=0.7609
