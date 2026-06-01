# CATF-v2 Curve Diagnosis

Generated: `2026-06-01T13:08:26`

No training was run. This report compares existing clean native and CATF-v2 `results.csv` curves.

## Seed 0

- First divergence epochs: `{'precision': 1, 'recall': 1, 'map50': 1, 'map50_95': 2}`.
- Feedback epochs: `[5, 10, 15, 20, 25, 30, 35, 40, 45]`.
- First freeze epoch: `40`; CATF raw metric change from first freeze to final: `{'precision': -0.0017300000000000093, 'recall': 0.038470000000000004, 'map50': 0.030890000000000084, 'map50_95': 0.03083999999999998}`.
- Final delta P/R/mAP50/mAP50-95: `{'precision': -0.0417, 'recall': 0.0218, 'map50': 0.0223, 'map50_95': 0.0264}`.

| feedback epoch | action | active classes | ΔP at epoch | ΔR at epoch | ΔmAP50 at epoch | ΔmAP50-95 at epoch | next-window ΔP change | next-window ΔR change |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 5 | shrink | [] | +0.0080 | -0.1018 | -0.0357 | -0.0083 | +0.0148 | +0.1616 |
| 10 | accept | [11] | +0.0228 | +0.0598 | +0.0268 | -0.0060 | +0.1578 | -0.1702 |
| 15 | observe | [] | +0.1806 | -0.1104 | +0.0077 | -0.0087 | -0.1854 | +0.1587 |
| 20 | observe | [] | -0.0048 | +0.0483 | +0.0561 | +0.0247 | +0.0322 | -0.0231 |
| 25 | observe | [] | +0.0274 | +0.0252 | +0.0534 | +0.0078 | -0.1133 | +0.0242 |
| 30 | shrink | [] | -0.0859 | +0.0494 | -0.0243 | -0.0019 | +0.1864 | +0.0067 |
| 35 | observe | [] | +0.1005 | +0.0561 | +0.0647 | +0.0541 | -0.1211 | +0.0517 |
| 40 | freeze | [] | -0.0206 | +0.1078 | +0.0157 | +0.0161 | +0.1004 | -0.1294 |
| 45 | freeze | [] | +0.0798 | -0.0216 | +0.0412 | +0.0274 | -0.0536 | +0.0478 |

## Seed 1

- First divergence epochs: `{'precision': 2, 'recall': 3, 'map50': 1, 'map50_95': 2}`.
- Feedback epochs: `[5, 10, 15, 20, 25, 30, 35, 40, 45]`.
- First freeze epoch: `40`; CATF raw metric change from first freeze to final: `{'precision': 0.0958199999999999, 'recall': -0.033279999999999976, 'map50': 0.055850000000000066, 'map50_95': 0.028260000000000007}`.
- Final delta P/R/mAP50/mAP50-95: `{'precision': -0.0034, 'recall': 0.0474, 'map50': 0.0007, 'map50_95': 0.0099}`.

| feedback epoch | action | active classes | ΔP at epoch | ΔR at epoch | ΔmAP50 at epoch | ΔmAP50-95 at epoch | next-window ΔP change | next-window ΔR change |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 5 | shrink | [] | +0.0071 | -0.0544 | -0.0359 | -0.0159 | +0.0854 | -0.0704 |
| 10 | shrink | [] | +0.0924 | -0.1248 | -0.0779 | -0.0372 | -0.1542 | +0.1785 |
| 15 | shrink | [] | -0.0618 | +0.0538 | -0.0085 | +0.0061 | +0.1168 | -0.1183 |
| 20 | shrink | [] | +0.0550 | -0.0646 | -0.0212 | -0.0399 | -0.1892 | +0.1351 |
| 25 | shrink | [] | -0.1342 | +0.0705 | +0.0228 | +0.0104 | +0.0397 | -0.0601 |
| 30 | shrink | [] | -0.0945 | +0.0105 | -0.0377 | -0.0395 | +0.0345 | +0.0242 |
| 35 | shrink | [] | -0.0599 | +0.0347 | -0.0013 | -0.0049 | +0.0479 | +0.0776 |
| 40 | freeze | [] | -0.0121 | +0.1123 | +0.0342 | +0.0260 | +0.0640 | -0.1183 |
| 45 | freeze | [] | +0.0519 | -0.0060 | +0.0436 | +0.0378 | +0.0154 | -0.0244 |

## Seed 2

- First divergence epochs: `{'precision': 1, 'recall': 4, 'map50': 3, 'map50_95': 3}`.
- Feedback epochs: `[5, 10, 15, 20, 25, 30, 35, 40, 45]`.
- First freeze epoch: `40`; CATF raw metric change from first freeze to final: `{'precision': -0.1057499999999999, 'recall': 0.06373000000000006, 'map50': -0.017160000000000064, 'map50_95': -0.021440000000000015}`.
- Final delta P/R/mAP50/mAP50-95: `{'precision': 0.1395, 'recall': -0.1247, 'map50': -0.0269, 'map50_95': -0.0285}`.

| feedback epoch | action | active classes | ΔP at epoch | ΔR at epoch | ΔmAP50 at epoch | ΔmAP50-95 at epoch | next-window ΔP change | next-window ΔR change |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 5 | shrink | [] | -0.0184 | -0.0721 | -0.0366 | -0.0204 | -0.0323 | +0.0569 |
| 10 | shrink | [] | -0.0507 | -0.0152 | +0.0025 | -0.0061 | +0.0105 | -0.0253 |
| 15 | shrink | [] | -0.0402 | -0.0405 | -0.0343 | -0.0132 | -0.1573 | +0.1259 |
| 20 | shrink | [] | -0.1975 | +0.0855 | -0.0460 | -0.0293 | +0.2756 | -0.1362 |
| 25 | accept | [8] | +0.0780 | -0.0507 | -0.0085 | +0.0028 | -0.0103 | -0.0069 |
| 30 | observe | [] | +0.0678 | -0.0577 | +0.0240 | +0.0415 | -0.0895 | +0.0770 |
| 35 | shrink | [] | -0.0217 | +0.0193 | -0.0041 | +0.0057 | +0.0908 | -0.0949 |
| 40 | freeze | [] | +0.0691 | -0.0756 | -0.0118 | +0.0034 | +0.0425 | -0.0264 |
| 45 | freeze | [] | +0.1116 | -0.1020 | -0.0405 | -0.0468 | -0.1789 | +0.1039 |

## Curve-Level Conclusions

- Seed 0 Precision weakness is present from the beginning and is not corrected by CATF-v2; the later active class window does not explain the full Precision loss by itself.
- Seed 2 Recall loss is also an early/global trajectory shift rather than a simple direct result of the small ROI intervention at epoch 25.
- Freeze at epoch 40 prevents late corrective updates; after freeze the model can still drift through normal YOLO training, but CATF-v2 no longer adapts.
- ROI events are sparse and do not align strongly enough with the final global deltas to claim ROI augmentation as the primary cause of gains.
