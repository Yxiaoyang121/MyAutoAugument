# CATF-v2 Post-hoc Per-Class Threshold Calibration

Generated: `2026-06-01T15:12:05`

This is prediction-only analysis. No training was run.

The evaluator uses prediction JSON at `conf=0.10` and searches class thresholds from `0.10` to `0.70` in `0.05` steps.

## CATF-v2 Constrained Calibration Summary

| seed | P | R | mAP50 | mAP50-95 | ΔP vs clean | ΔR vs clean | ΔmAP50 vs clean | ΔmAP50-95 vs clean | constraint_failed | raise threshold | lower threshold |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|---|
| 0 | 0.6165 | 0.8089 | 0.7255 | 0.4753 | -0.0074 | +0.0444 | +0.0557 | +0.0352 | false | [] | ['6:漏背锡->0.10', '7:碰伤->0.10', '9:轮廓划伤->0.10', '10:锡丝残留->0.10', '11:锡尖->0.10'] |
| 1 | 0.5955 | 0.8299 | 0.7050 | 0.4566 | +0.0282 | +0.0569 | +0.0227 | +0.0120 | false | [] | ['1:OK3->0.10', '5:浅划伤->0.10', '6:漏背锡->0.10', '7:碰伤->0.10', '8:脏污->0.10', '9:轮廓划伤->0.10', '12:锡膏->0.10'] |
| 2 | 0.5944 | 0.7388 | 0.6253 | 0.4020 | -0.0049 | -0.0175 | -0.0608 | -0.0579 | true | [] | ['1:OK3->0.10', '4:油污->0.10', '6:漏背锡->0.10', '7:碰伤->0.10', '8:脏污->0.10', '9:轮廓划伤->0.10', '11:锡尖->0.10', '12:锡膏->0.10'] |

## Objective-Specific Threshold Recommendations

- `constrained_score` maximizes Recall while keeping P/mAP within the clean-reference tolerance.
- `balanced_score` and `industrial_score` are more useful when the operator wants an explicit Precision/Recall balance.

### Seed 0
- `constrained_score`: P=0.6165, R=0.8089, mAP50=0.7255, mAP50-95=0.4753, constraint_failed=false, raise=[], lower=['6:漏背锡->0.10', '7:碰伤->0.10', '9:轮廓划伤->0.10', '10:锡丝残留->0.10', '11:锡尖->0.10']
- `balanced_score`: P=0.6833, R=0.7960, mAP50=0.7163, mAP50-95=0.4710, constraint_failed=false, raise=['0:OK2->0.60', '1:OK3->0.70', '2:加强筋打伤->0.45', '3:开裂->0.60', '4:油污->0.30', '5:浅划伤->0.50', '8:脏污->0.30'], lower=['6:漏背锡->0.10', '7:碰伤->0.10', '9:轮廓划伤->0.10', '10:锡丝残留->0.10', '11:锡尖->0.20']
- `industrial_score`: P=0.6833, R=0.7960, mAP50=0.7163, mAP50-95=0.4710, constraint_failed=false, raise=['0:OK2->0.60', '1:OK3->0.70', '2:加强筋打伤->0.45', '3:开裂->0.60', '4:油污->0.30', '5:浅划伤->0.50', '8:脏污->0.30'], lower=['6:漏背锡->0.10', '7:碰伤->0.10', '9:轮廓划伤->0.10', '10:锡丝残留->0.10', '11:锡尖->0.20']

### Seed 1
- `constrained_score`: P=0.5955, R=0.8299, mAP50=0.7050, mAP50-95=0.4566, constraint_failed=false, raise=[], lower=['1:OK3->0.10', '5:浅划伤->0.10', '6:漏背锡->0.10', '7:碰伤->0.10', '8:脏污->0.10', '9:轮廓划伤->0.10', '12:锡膏->0.10']
- `balanced_score`: P=0.6982, R=0.8093, mAP50=0.6934, mAP50-95=0.4534, constraint_failed=false, raise=['0:OK2->0.55', '1:OK3->0.40', '2:加强筋打伤->0.30', '3:开裂->0.35', '5:浅划伤->0.55', '11:锡尖->0.30'], lower=['6:漏背锡->0.10', '7:碰伤->0.20', '8:脏污->0.15', '9:轮廓划伤->0.10', '12:锡膏->0.10']
- `industrial_score`: P=0.6982, R=0.8093, mAP50=0.6934, mAP50-95=0.4534, constraint_failed=false, raise=['0:OK2->0.55', '1:OK3->0.40', '2:加强筋打伤->0.30', '3:开裂->0.35', '5:浅划伤->0.55', '11:锡尖->0.30'], lower=['6:漏背锡->0.10', '7:碰伤->0.20', '8:脏污->0.15', '9:轮廓划伤->0.10', '12:锡膏->0.10']

### Seed 2
- `constrained_score`: P=0.5944, R=0.7388, mAP50=0.6253, mAP50-95=0.4020, constraint_failed=true, raise=[], lower=['1:OK3->0.10', '4:油污->0.10', '6:漏背锡->0.10', '7:碰伤->0.10', '8:脏污->0.10', '9:轮廓划伤->0.10', '11:锡尖->0.10', '12:锡膏->0.10']
- `balanced_score`: P=0.7384, R=0.6981, mAP50=0.6003, mAP50-95=0.3888, constraint_failed=true, raise=['0:OK2->0.70', '1:OK3->0.70', '3:开裂->0.50', '4:油污->0.45', '5:浅划伤->0.50', '6:漏背锡->0.50', '8:脏污->0.35', '9:轮廓划伤->0.40', '12:锡膏->0.55'], lower=['7:碰伤->0.10', '11:锡尖->0.10']
- `industrial_score`: P=0.7255, R=0.7045, mAP50=0.6045, mAP50-95=0.3908, constraint_failed=true, raise=['0:OK2->0.70', '1:OK3->0.70', '3:开裂->0.50', '4:油污->0.45', '5:浅划伤->0.50', '6:漏背锡->0.50', '8:脏污->0.35', '12:锡膏->0.55'], lower=['7:碰伤->0.10', '9:轮廓划伤->0.10', '11:锡尖->0.10']


## Answers

- Seed 0 Precision repair: `yes`.
- Seed 2 Recall repair: `no`.
- CATF-v2 pass count after constrained calibration: `2/3`.
- Threshold calibration is best treated as a post-processing/deployment layer, not evidence that training augmentation itself is stable.
- If seed 2 still fails Recall after calibration, its failure is training/trajectory degradation rather than a pure threshold issue.
