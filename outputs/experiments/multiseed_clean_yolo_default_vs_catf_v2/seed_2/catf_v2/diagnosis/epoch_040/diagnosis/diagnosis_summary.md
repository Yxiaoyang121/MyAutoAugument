# Validation Error Diagnosis

- Status: completed
- TP: 715
- FP: 305
- FN: 173
- Precision: 0.7010
- Recall: 0.7901

## Diagnosis Vector
- small_object_score: 0.2649
- low_contrast_score: 0.5324
- class_imbalance_score: 0.8591
- localization_score: 0.0188
- false_positive_score: 0.2990

## Issues
- low_contrast_missed_defect severity=high suggestions=contrast, gamma, clahe, brightness, sharpen
- low_contrast_missed_defect severity=medium suggestions=contrast, gamma, clahe, brightness, sharpen
- class_imbalance severity=medium suggestions=class-aware-sampling, targeted-augmentation, class-balanced-policy
- high_false_positive severity=medium suggestions=hard-negative-review, reduce-noise, reduce-blur, mild-lighting
- localization_bias severity=medium suggestions=mild-scale, mild-translate, reduce-rotate, reduce-shear, reduce-perspective

## Per-Class Summary
- OK2: gt=64 tp=64 fp=22 fn=0 precision=0.7442 recall=1.0000
- OK3: gt=274 tp=273 fp=72 fn=1 precision=0.7913 recall=0.9964
- 加强筋打伤: gt=8 tp=7 fp=4 fn=1 precision=0.6364 recall=0.8750
- 开裂: gt=2 tp=2 fp=5 fn=0 precision=0.2857 recall=1.0000
- 油污: gt=18 tp=7 fp=18 fn=10 precision=0.2800 recall=0.3889
- 浅划伤: gt=13 tp=8 fp=12 fn=4 precision=0.4000 recall=0.6154
- 漏背锡: gt=46 tp=29 fp=16 fn=13 precision=0.6444 recall=0.6304
- 碰伤: gt=269 tp=184 fp=64 fn=80 precision=0.7419 recall=0.6840
- 脏污: gt=42 tp=29 fp=43 fn=9 precision=0.4028 recall=0.6905
- 轮廓划伤: gt=84 tp=48 fp=17 fn=35 precision=0.7385 recall=0.5714
- 锡丝残留: gt=12 tp=9 fp=4 fn=3 precision=0.6923 recall=0.7500
- 锡尖: gt=27 tp=23 fp=5 fn=4 precision=0.8214 recall=0.8519
- 锡膏: gt=46 tp=32 fp=23 fn=13 precision=0.5818 recall=0.6957
