# Seed2 Curve Degradation Analysis

No training was run. This report compares existing `results.csv` curves.

## Degradation Onset

| run | metric | first negative | first <= -0.005 | first <= -0.010 | final epoch delta |
|---|---|---:|---:|---:|---:|
| fixed CATF-v2 | precision | 7 | 7 | 7 | -0.0003 |
| fixed CATF-v2 | recall | 6 | 6 | 6 | +0.0095 |
| fixed CATF-v2 | map50 | 8 | 8 | 8 | -0.0206 |
| fixed CATF-v2 | map50_95 | 6 | 8 | 8 | -0.0219 |
| Gated | precision | 7 | 7 | 7 | -0.0641 |
| Gated | recall | 6 | 6 | 6 | +0.0288 |
| Gated | map50 | 8 | 8 | 8 | -0.0726 |
| Gated | map50_95 | 6 | 8 | 8 | -0.0595 |

## Feedback Epoch Context

| epoch | fixed action | fixed active classes | dP | dR | dM50 | dM95 |
|---:|---|---|---:|---:|---:|---:|
| 5 | accept | [9] | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| 10 | shrink | [] | +0.0368 | -0.0452 | -0.0208 | -0.0181 |
| 15 | shrink | [] | -0.0524 | +0.0268 | +0.0055 | -0.0007 |
| 20 | shrink | [] | -0.0137 | +0.0013 | -0.0119 | -0.0016 |
| 25 | accept | [11] | +0.0496 | -0.0299 | +0.0507 | +0.0524 |
| 30 | shrink | [11] | -0.0105 | +0.0485 | +0.0476 | +0.0411 |
| 35 | accept | [8, 11] | +0.0290 | +0.0279 | +0.0212 | +0.0167 |
| 40 | freeze | [] | -0.0141 | -0.0061 | -0.0015 | +0.0052 |
| 45 | freeze | [] | +0.0081 | -0.0379 | -0.0306 | -0.0376 |
| 50 |  | [] | -0.0003 | +0.0095 | -0.0206 | -0.0219 |

## Answers

- Recall first lags clean at epoch `6`; the first >=0.01 lag is also epoch `6`.
- mAP50 first clearly lags clean at epoch `8`.
- mAP50-95 first turns negative at epoch `6` and clearly lags at epoch `8`.
- The only policy update before that window is epoch 5: class 9 is activated with `sharpen_mild` and `local_contrast`.
- Gated seed2 falls back at epoch 10, after epochs 6-10 have already run under the candidate policy; this explains why the fallback was too late to preserve clean final behavior.
