# Seed2 Root Cause Summary

No training was run. The audit combines existing curves, final metrics, policy history, augmentation stats, and prediction-only diffs.

## Most Likely Sources

1. fallback/gate too late after epoch-5 candidate activation: Recall first lags at epoch 6 and AP metrics clearly lag at epoch 8, before the epoch-10 gate can react.
2. class-9 texture ROI intervention: The only image-space change in the first degradation window is class 9 with sharpen_mild/local_contrast; class 9 later has large Recall/AP regression and 25 ROI applications.
3. missing causal validation for active issue attribution: Class 9 diagnosis confidence is high, but downstream metrics move in the precision-up/recall-down direction; CP-CATF is needed before image-space intervention.
4. non-active class regression: Classes without ROI application also lose AP or Recall, so the side effect is not only direct class-9 harm.
5. strong clean seed2 baseline should not be disturbed: Safe and adaptive-RB strict no-op exactly reproduce clean seed2; fixed/gated do not.

## Direct Answers

- Most suspicious policy update: `epoch 5 accept/propose class 9 texture_boundary_weak with sharpen_mild and local_contrast`.
- Most suspicious active class: `class 9`.
- Most suspicious op family: `ROI texture ops as a pair: sharpen_mild + local_contrast; current logs cannot isolate one op`.
- Non-active class regression exists: `true`.
- Recommend seed2 default no-op: `true`.
- Recommend sampler-only before image ops: `true`.
- Recommend disabling texture ops without ablation: `false`.
- Recommend CP-CATF causal probe: `true`.

## Worst Injured Classes

| dimension | classes |
|---|---|
| recall_drop | 5:浅划伤 (dR=-0.2704, dAP50=+0.1072, dAP95=+0.1010, active=false, ROI=0), 9:轮廓划伤 (dR=-0.1667, dAP50=-0.0466, dAP95=-0.0654, active=true, ROI=25), 2:加强筋打伤 (dR=-0.1250, dAP50=+0.0045, dAP95=-0.0463, active=false, ROI=0), 6:漏背锡 (dR=-0.0745, dAP50=-0.0370, dAP95=+0.0196, active=false, ROI=0), 12:锡膏 (dR=-0.0044, dAP50=-0.0483, dAP95=-0.0417, active=false, ROI=0) |
| ap50_drop | 10:锡丝残留 (dR=+0.0000, dAP50=-0.0748, dAP95=-0.0880, active=false, ROI=0), 8:脏污 (dR=+0.0301, dAP50=-0.0559, dAP95=-0.0502, active=true, ROI=6), 12:锡膏 (dR=-0.0044, dAP50=-0.0483, dAP95=-0.0417, active=false, ROI=0), 9:轮廓划伤 (dR=-0.1667, dAP50=-0.0466, dAP95=-0.0654, active=true, ROI=25), 6:漏背锡 (dR=-0.0745, dAP50=-0.0370, dAP95=+0.0196, active=false, ROI=0) |
| ap50_95_drop | 3:开裂 (dR=+0.0000, dAP50=+0.0000, dAP95=-0.1492, active=false, ROI=0), 10:锡丝残留 (dR=+0.0000, dAP50=-0.0748, dAP95=-0.0880, active=false, ROI=0), 9:轮廓划伤 (dR=-0.1667, dAP50=-0.0466, dAP95=-0.0654, active=true, ROI=25), 8:脏污 (dR=+0.0301, dAP50=-0.0559, dAP95=-0.0502, active=true, ROI=6), 2:加强筋打伤 (dR=-0.1250, dAP50=+0.0045, dAP95=-0.0463, active=false, ROI=0) |
| fn_increase | 9:轮廓划伤 (dR=-0.1667, dAP50=-0.0466, dAP95=-0.0654, active=true, ROI=25), 5:浅划伤 (dR=-0.2704, dAP50=+0.1072, dAP95=+0.1010, active=false, ROI=0), 6:漏背锡 (dR=-0.0745, dAP50=-0.0370, dAP95=+0.0196, active=false, ROI=0), 2:加强筋打伤 (dR=-0.1250, dAP50=+0.0045, dAP95=-0.0463, active=false, ROI=0), 12:锡膏 (dR=-0.0044, dAP50=-0.0483, dAP95=-0.0417, active=false, ROI=0) |
| actionable_overlap | 8:脏污 (dR=+0.0301, dAP50=-0.0559, dAP95=-0.0502, active=true, ROI=6), 9:轮廓划伤 (dR=-0.1667, dAP50=-0.0466, dAP95=-0.0654, active=true, ROI=25) |

## Minimal Ablation Plan

| priority | ablation | target | run length | expected answer |
|---:|---|---|---|---|
| 1 | CP-CATF causal probe before augmentation | verify whether class-9 texture issue has causal positive evidence before any image-space op | prediction/diagnosis only, then 5-10ep if needed | decide whether class 9 should enter candidate branch at all |
| 2 | seed2 no industrial aug / strict no-op control | confirm reproducibility of clean behavior in the same custom trainer path | already covered by Safe/adaptive-RB; no new 50ep needed | baseline for any future short ablation |
| 3 | seed2 texture ops disabled for class 9 | test whether the epoch5 class-9 image-space intervention is the causal pollutant | 10ep short train first; 50ep only if early curve is clean | if epoch6-10 no longer lags, class-9 texture intervention is implicated |
| 4 | seed2 only roi_sharpen_mild vs only roi_local_contrast | separate the two texture ops that are currently co-applied | paired 10ep short trains | identify whether one op or their combination drives recall/mAP loss |
| 5 | seed2 sampler_only | test sample-aware routing without changing pixels | 10ep short train; 50ep only if stable | separate sampling effects from ROI image perturbation |
| 6 | seed2 no roi_sharpen_mild / no roi_local_contrast | operator family confirmation if only-op runs are noisy | 10ep short train | confirm which removal restores early Recall/mAP |
| 7 | disable active class 9, then allow later class 11/8 only | test whether later active classes are safe after skipping the early pollutant | 10ep to epoch15/20; 50ep only if pass | separate early class9 failure from later policy updates |

## Verification

- No training run: `true`.
- Predict-only validation was used only to generate clean/fixed best prediction JSON for error comparison.
- Missing tests skipped: `['tests/test_catf_v2_causal_probe.py', 'tests/test_catf_v2_adaptive_rb_v2.py']`.
- Pytest result: `111 passed`.
