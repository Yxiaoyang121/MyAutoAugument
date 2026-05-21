# Counterfactual Diagnosis Summary

- Scope: prediction-only counterfactual diagnosis; no training was run.
- Baseline FN count: `215`
- Tested FN count: `200`
- Highest recovery transform: `sharpen_mild` rate=`0.0700`
- diag_policy_001 photometric unique FN recovery rate: `0.1050`

## Transform Recovery

| transform | tested FN | recovered | recovery_rate | main affected classes |
|---|---:|---:|---:|---|
| `sharpen_mild` | 200 | 14 | 0.0700 | 碰伤, 脏污, OK3, 油污, 轮廓划伤 |
| `contrast_up_130` | 200 | 12 | 0.0600 | 碰伤, 脏污, OK3, 加强筋打伤, 轮廓划伤 |
| `contrast_up_115` | 200 | 10 | 0.0500 | 碰伤, OK3, 脏污, 轮廓划伤, 锡尖 |
| `clahe_clip2` | 200 | 8 | 0.0400 | 碰伤, OK3, 脏污, 油污 |
| `combined_photometric` | 200 | 8 | 0.0400 | 碰伤, OK3, 脏污 |
| `clahe_clip3` | 200 | 6 | 0.0300 | OK3, 碰伤, 脏污 |
| `gamma_brighten_085` | 200 | 5 | 0.0250 | 碰伤, 漏背锡, 脏污 |
| `brightness_up_30` | 200 | 4 | 0.0200 | 碰伤, 漏背锡 |
| `gamma_brighten_075` | 200 | 4 | 0.0200 | 碰伤, 脏污 |
| `brightness_up_15` | 200 | 3 | 0.0150 | 碰伤 |
| `zoom_in_context` | 200 | 2 | 0.0100 | OK3, 脏污 |
