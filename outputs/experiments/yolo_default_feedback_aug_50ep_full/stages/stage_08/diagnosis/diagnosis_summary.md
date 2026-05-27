# Validation Error Diagnosis

- Status: completed
- TP: 696
- FP: 287
- FN: 186
- Precision: 0.7080
- Recall: 0.7691

## Diagnosis Vector
- small_object_score: 0.2903
- low_contrast_score: 0.5317
- class_imbalance_score: 0.9175
- localization_score: 0.0254
- false_positive_score: 0.2920

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=20 fn=0 precision=0.7619 recall=1.0000
- OK3: gt=274 tp=274 fp=67 fn=0 precision=0.8035 recall=1.0000
- 加强筋打伤: gt=8 tp=7 fp=0 fn=1 precision=1.0000 recall=0.8750
- 开裂: gt=2 tp=2 fp=3 fn=0 precision=0.4000 recall=1.0000
- 油污: gt=18 tp=4 fp=15 fn=13 precision=0.2105 recall=0.2222
- 浅划伤: gt=13 tp=6 fp=1 fn=7 precision=0.8571 recall=0.4615
- 漏背锡: gt=46 tp=25 fp=25 fn=13 precision=0.5000 recall=0.5435
- 碰伤: gt=269 tp=181 fp=85 fn=80 precision=0.6805 recall=0.6729
- 脏污: gt=42 tp=19 fp=20 fn=22 precision=0.4872 recall=0.4524
- 轮廓划伤: gt=84 tp=49 fp=28 fn=31 precision=0.6364 recall=0.5833
- 锡丝残留: gt=12 tp=9 fp=11 fn=3 precision=0.4500 recall=0.7500
- 锡尖: gt=27 tp=24 fp=3 fn=3 precision=0.8889 recall=0.8889
- 锡膏: gt=46 tp=32 fp=9 fn=13 precision=0.7805 recall=0.6957
