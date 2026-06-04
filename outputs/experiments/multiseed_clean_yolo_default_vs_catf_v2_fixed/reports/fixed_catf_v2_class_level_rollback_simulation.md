# Fixed CATF-v2 Class-Level Rollback Simulation

No training was run. This is a post-hoc class-level decision analysis comparing clean native, old CATF-v2, fixed CATF-v2, and selected threshold calibration.

## keep_fixed_catf_v2

4:油污, 11:锡尖

## threshold_calibration_only

8:脏污

## rollback_candidate

2:加强筋打伤, 10:锡丝残留

## high_risk_negative_effect

9:轮廓划伤, 12:锡膏

## unstable_not_mandatory_rollback

0:OK2, 1:OK3, 3:开裂, 5:浅划伤, 6:漏背锡, 7:碰伤

## Per-Class Evidence

| class | action | negative seeds | rollback seeds | threshold-help seeds |
|---|---|---:|---:|---:|
| 0:OK2 | unstable_not_mandatory_rollback | 1 | 0 | 3 |
| 1:OK3 | unstable_not_mandatory_rollback | 1 | 1 | 2 |
| 2:加强筋打伤 | rollback_candidate | 2 | 2 | 1 |
| 3:开裂 | unstable_not_mandatory_rollback | 1 | 1 | 1 |
| 4:油污 | keep_fixed_catf_v2 | 0 | 0 | 2 |
| 5:浅划伤 | unstable_not_mandatory_rollback | 1 | 0 | 3 |
| 6:漏背锡 | unstable_not_mandatory_rollback | 1 | 0 | 3 |
| 7:碰伤 | unstable_not_mandatory_rollback | 1 | 0 | 3 |
| 8:脏污 | threshold_calibration_only | 2 | 0 | 3 |
| 9:轮廓划伤 | high_risk_negative_effect | 2 | 1 | 1 |
| 10:锡丝残留 | rollback_candidate | 2 | 2 | 0 |
| 11:锡尖 | keep_fixed_catf_v2 | 0 | 0 | 3 |
| 12:锡膏 | high_risk_negative_effect | 3 | 1 | 1 |
