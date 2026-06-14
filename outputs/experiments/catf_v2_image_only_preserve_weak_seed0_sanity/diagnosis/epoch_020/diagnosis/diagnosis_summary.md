# Validation Error Diagnosis

- Status: completed
- TP: 692
- FP: 419
- FN: 190
- Precision: 0.6229
- Recall: 0.7646

## Diagnosis Vector
- small_object_score: 0.3073
- low_contrast_score: 0.5684
- class_imbalance_score: 0.9953
- localization_score: 0.0254
- false_positive_score: 0.3771

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=30 fn=0 precision=0.6809 recall=1.0000
- OK3: gt=274 tp=272 fp=70 fn=2 precision=0.7953 recall=0.9927
- 加强筋打伤: gt=8 tp=6 fp=0 fn=2 precision=1.0000 recall=0.7500
- 开裂: gt=2 tp=0 fp=0 fn=2 precision=0.0000 recall=0.0000
- 油污: gt=18 tp=6 fp=45 fn=10 precision=0.1176 recall=0.3333
- 浅划伤: gt=13 tp=6 fp=4 fn=7 precision=0.6000 recall=0.4615
- 漏背锡: gt=46 tp=36 fp=64 fn=6 precision=0.3600 recall=0.7826
- 碰伤: gt=269 tp=170 fp=93 fn=96 precision=0.6464 recall=0.6320
- 脏污: gt=42 tp=19 fp=24 fn=19 precision=0.4419 recall=0.4524
- 轮廓划伤: gt=84 tp=43 fp=28 fn=36 precision=0.6056 recall=0.5119
- 锡丝残留: gt=12 tp=9 fp=10 fn=3 precision=0.4737 recall=0.7500
- 锡尖: gt=27 tp=21 fp=4 fn=3 precision=0.8400 recall=0.7778
- 锡膏: gt=46 tp=40 fp=47 fn=4 precision=0.4598 recall=0.8696
