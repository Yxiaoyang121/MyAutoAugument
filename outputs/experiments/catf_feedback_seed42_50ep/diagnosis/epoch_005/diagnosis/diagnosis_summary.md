# Validation Error Diagnosis

- Status: completed
- TP: 634
- FP: 639
- FN: 247
- Precision: 0.4980
- Recall: 0.7006

## Diagnosis Vector
- small_object_score: 0.3633
- low_contrast_score: 0.4692
- class_imbalance_score: 0.9953
- localization_score: 0.0265
- false_positive_score: 0.5020

## Issues
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=19 fn=0 precision=0.7711 recall=1.0000
- OK3: gt=274 tp=273 fp=76 fn=1 precision=0.7822 recall=0.9964
- 加强筋打伤: gt=8 tp=0 fp=0 fn=8 precision=0.0000 recall=0.0000
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=4 fp=50 fn=14 precision=0.0741 recall=0.2222
- 浅划伤: gt=13 tp=0 fp=0 fn=13 precision=0.0000 recall=0.0000
- 漏背锡: gt=46 tp=41 fp=258 fn=2 precision=0.1371 recall=0.8913
- 碰伤: gt=269 tp=171 fp=159 fn=89 precision=0.5182 recall=0.6357
- 脏污: gt=42 tp=6 fp=11 fn=35 precision=0.3529 recall=0.1429
- 轮廓划伤: gt=84 tp=38 fp=45 fn=36 precision=0.4578 recall=0.4524
- 锡丝残留: gt=12 tp=0 fp=0 fn=12 precision=0.0000 recall=0.0000
- 锡尖: gt=27 tp=17 fp=14 fn=10 precision=0.5484 recall=0.6296
- 锡膏: gt=46 tp=20 fp=7 fn=25 precision=0.7407 recall=0.4348
