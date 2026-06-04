# Validation Error Diagnosis

- Status: completed
- TP: 715
- FP: 359
- FN: 170
- Precision: 0.6657
- Recall: 0.7901

## Diagnosis Vector
- small_object_score: 0.2666
- low_contrast_score: 0.5382
- class_imbalance_score: 0.9953
- localization_score: 0.0221
- false_positive_score: 0.3343

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=19 fn=0 precision=0.7711 recall=1.0000
- OK3: gt=274 tp=273 fp=68 fn=1 precision=0.8006 recall=0.9964
- 加强筋打伤: gt=8 tp=4 fp=0 fn=3 precision=1.0000 recall=0.5000
- 开裂: gt=2 tp=0 fp=6 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=8 fp=36 fn=10 precision=0.1818 recall=0.4444
- 浅划伤: gt=13 tp=1 fp=0 fn=10 precision=1.0000 recall=0.0769
- 漏背锡: gt=46 tp=35 fp=38 fn=4 precision=0.4795 recall=0.7609
- 碰伤: gt=269 tp=198 fp=97 fn=65 precision=0.6712 recall=0.7361
- 脏污: gt=42 tp=23 fp=35 fn=17 precision=0.3966 recall=0.5476
- 轮廓划伤: gt=84 tp=45 fp=28 fn=37 precision=0.6164 recall=0.5357
- 锡丝残留: gt=12 tp=9 fp=6 fn=3 precision=0.6000 recall=0.7500
- 锡尖: gt=27 tp=22 fp=8 fn=5 precision=0.7333 recall=0.8148
- 锡膏: gt=46 tp=33 fp=18 fn=13 precision=0.6471 recall=0.7174
