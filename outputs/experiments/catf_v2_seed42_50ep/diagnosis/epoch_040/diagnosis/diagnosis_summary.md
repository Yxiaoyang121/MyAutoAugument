# Validation Error Diagnosis

- Status: completed
- TP: 718
- FP: 331
- FN: 174
- Precision: 0.6845
- Recall: 0.7934

## Diagnosis Vector
- small_object_score: 0.2632
- low_contrast_score: 0.5431
- class_imbalance_score: 0.8203
- localization_score: 0.0144
- false_positive_score: 0.3155

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=25 fn=0 precision=0.7191 recall=1.0000
- OK3: gt=274 tp=270 fp=68 fn=4 precision=0.7988 recall=0.9854
- 加强筋打伤: gt=8 tp=7 fp=3 fn=1 precision=0.7000 recall=0.8750
- 开裂: gt=2 tp=2 fp=4 fn=0 precision=0.3333 recall=1.0000
- 油污: gt=18 tp=9 fp=33 fn=9 precision=0.2143 recall=0.5000
- 浅划伤: gt=13 tp=7 fp=7 fn=6 precision=0.5000 recall=0.5385
- 漏背锡: gt=46 tp=35 fp=30 fn=7 precision=0.5385 recall=0.7609
- 碰伤: gt=269 tp=184 fp=85 fn=80 precision=0.6840 recall=0.6840
- 脏污: gt=42 tp=23 fp=21 fn=19 precision=0.5227 recall=0.5476
- 轮廓划伤: gt=84 tp=51 fp=30 fn=29 precision=0.6296 recall=0.6071
- 锡丝残留: gt=12 tp=10 fp=9 fn=2 precision=0.5263 recall=0.8333
- 锡尖: gt=27 tp=27 fp=6 fn=0 precision=0.8182 recall=1.0000
- 锡膏: gt=46 tp=29 fp=10 fn=17 precision=0.7436 recall=0.6304
