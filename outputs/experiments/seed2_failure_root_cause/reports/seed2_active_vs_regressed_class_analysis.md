# Seed2 Active vs Regressed Class Analysis

## Active Classes

| class | ROI applied | dominant issue(s) | dR | dAP50 | dAP95 | final improved? |
|---|---:|---|---:|---:|---:|:---:|
| 8:脏污 | 6 | ['texture_boundary_weak', 'high_fp', 'high_fp', 'high_fp', 'high_fp', 'weak_localization', 'high_fp', 'high_fp', 'high_fp'] | +0.0301 | -0.0559 | -0.0502 | false |
| 9:轮廓划伤 | 25 | ['high_fp', 'high_fp', 'high_fp', 'high_fp', 'high_fp', 'high_fp', 'high_fp', 'high_fp', 'texture_boundary_weak'] | -0.1667 | -0.0466 | -0.0654 | false |
| 11:锡尖 | 59 | ['texture_boundary_weak', 'texture_boundary_weak', 'low_contrast_fn', 'weak_localization', 'low_contrast_fn', 'low_contrast_fn', 'texture_boundary_weak', 'stable_class', 'high_fp'] | +0.0046 | +0.0042 | +0.0133 | true |

## Non-Active Regressions

| class | active | ROI | dR | dAP50 | dAP95 |
|---|:---:|---:|---:|---:|---:|
| 5:浅划伤 | false | 0 | -0.2704 | +0.1072 | +0.1010 |
| 3:开裂 | false | 0 | +0.0000 | +0.0000 | -0.1492 |
| 2:加强筋打伤 | false | 0 | -0.1250 | +0.0045 | -0.0463 |
| 10:锡丝残留 | false | 0 | +0.0000 | -0.0748 | -0.0880 |
| 6:漏背锡 | false | 0 | -0.0745 | -0.0370 | +0.0196 |
| 12:锡膏 | false | 0 | -0.0044 | -0.0483 | -0.0417 |
| 7:碰伤 | false | 0 | +0.0260 | -0.0243 | -0.0077 |

## Answers

- Active and regressed classes partially overlap: class 9 is the strongest overlap; class 8 loses AP but not Recall; class 11 improves.
- There is clear non-active regression: several classes with no ROI application lose AP50/AP50-95 or Recall.
- The epoch 5 class 9 activation was plausible from diagnosis confidence, but final effect was negative; this points to missing causal validation before image-space intervention.
