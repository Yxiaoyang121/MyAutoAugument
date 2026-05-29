# Validation Error Diagnosis

- Status: completed
- TP: 691
- FP: 278
- FN: 204
- Precision: 0.7131
- Recall: 0.7635

## Diagnosis Vector
- small_object_score: 0.3209
- low_contrast_score: 0.5833
- class_imbalance_score: 0.8786
- localization_score: 0.0110
- false_positive_score: 0.2869

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=20 fn=0 precision=0.7619 recall=1.0000
- OK3: gt=274 tp=274 fp=69 fn=0 precision=0.7988 recall=1.0000
- 加强筋打伤: gt=8 tp=6 fp=0 fn=2 precision=1.0000 recall=0.7500
- 开裂: gt=2 tp=2 fp=3 fn=0 precision=0.4000 recall=1.0000
- 油污: gt=18 tp=6 fp=15 fn=12 precision=0.2857 recall=0.3333
- 浅划伤: gt=13 tp=6 fp=7 fn=6 precision=0.4615 recall=0.4615
- 漏背锡: gt=46 tp=33 fp=31 fn=11 precision=0.5156 recall=0.7174
- 碰伤: gt=269 tp=172 fp=62 fn=91 precision=0.7350 recall=0.6394
- 脏污: gt=42 tp=17 fp=8 fn=25 precision=0.6800 recall=0.4048
- 轮廓划伤: gt=84 tp=40 fp=33 fn=43 precision=0.5479 recall=0.4762
- 锡丝残留: gt=12 tp=10 fp=11 fn=2 precision=0.4762 recall=0.8333
- 锡尖: gt=27 tp=27 fp=8 fn=0 precision=0.7714 recall=1.0000
- 锡膏: gt=46 tp=34 fp=11 fn=12 precision=0.7556 recall=0.7391
