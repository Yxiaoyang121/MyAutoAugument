# Validation Error Diagnosis

- Status: completed
- TP: 725
- FP: 305
- FN: 156
- Precision: 0.7039
- Recall: 0.8011

## Diagnosis Vector
- small_object_score: 0.2547
- low_contrast_score: 0.5353
- class_imbalance_score: 0.8337
- localization_score: 0.0265
- false_positive_score: 0.2961

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=21 fn=0 precision=0.7529 recall=1.0000
- OK3: gt=274 tp=273 fp=65 fn=1 precision=0.8077 recall=0.9964
- 加强筋打伤: gt=8 tp=6 fp=1 fn=2 precision=0.8571 recall=0.7500
- 开裂: gt=2 tp=2 fp=9 fn=0 precision=0.1818 recall=1.0000
- 油污: gt=18 tp=9 fp=24 fn=9 precision=0.2727 recall=0.5000
- 浅划伤: gt=13 tp=6 fp=12 fn=7 precision=0.3333 recall=0.4615
- 漏背锡: gt=46 tp=33 fp=26 fn=10 precision=0.5593 recall=0.7174
- 碰伤: gt=269 tp=184 fp=76 fn=75 precision=0.7077 recall=0.6840
- 脏污: gt=42 tp=22 fp=12 fn=16 precision=0.6471 recall=0.5238
- 轮廓划伤: gt=84 tp=56 fp=38 fn=23 precision=0.5957 recall=0.6667
- 锡丝残留: gt=12 tp=9 fp=5 fn=3 precision=0.6429 recall=0.7500
- 锡尖: gt=27 tp=27 fp=4 fn=0 precision=0.8710 recall=1.0000
- 锡膏: gt=46 tp=34 fp=12 fn=10 precision=0.7391 recall=0.7391
