# Seed0 CP-CATF Precision-Risk Audit

## Scope

- No training was run for this audit.
- Clean seed0 was reused from the paper-mode baseline.
- CP-CATF seed0 was read from `outputs/experiments/cp_catf_paper_mode_execution_fixed_seed0_only/`.
- Prediction-level analysis uses fresh inference caches for final val and probe split. Final-val threshold calibration is marked leakage analysis only.

## Global Result

| metric | clean | CP-CATF | delta |
|---|---:|---:|---:|
| P | 0.7513 | 0.7399 | -0.0114 |
| R | 0.6763 | 0.6763 | +0.0000 |
| mAP50 | 0.7566 | 0.7593 | +0.0027 |
| mAP50-95 | 0.5114 | 0.5028 | -0.0086 |

- Constraint failed: `True`; reasons: `['precision_drop_gt_0.01']`.
- At confidence `0.25`, matched FP changed from `344` to `354` (`+10`), while TP changed `+3` and FN changed `-3`.

## Precision Drop Top Classes

| class | name | clean P | CP P | dP | clean FP | CP FP | dFP | clean TP | CP TP | dTP | clean FN | CP FN | dFN |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 浅划伤 | 0.8417 | 0.6937 | -0.1481 | 3 | 17 | +14 | 6 | 6 | +0 | 7 | 7 | +0 |
| 2 | 加强筋打伤 | 0.8542 | 0.7325 | -0.1217 | 1 | 4 | +3 | 7 | 7 | +0 | 1 | 1 | +0 |
| 6 | 漏背锡 | 0.6570 | 0.5515 | -0.1056 | 33 | 45 | +12 | 28 | 30 | +2 | 18 | 16 | -2 |
| 4 | 油污 | 0.4816 | 0.3852 | -0.0964 | 12 | 13 | +1 | 7 | 5 | -2 | 11 | 13 | +2 |
| 8 | 脏污 | 0.6744 | 0.6023 | -0.0721 | 16 | 22 | +6 | 25 | 25 | +0 | 17 | 17 | +0 |

## FP Increase Top Classes

| class | name | active | ROI affected | dFP | dTP | dFN | dAP50 | dAP50-95 |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 5 | 浅划伤 | `False` | `False` | +14 | +0 | +0 | -0.0108 | -0.0620 |
| 6 | 漏背锡 | `False` | `False` | +12 | +2 | -2 | -0.0259 | -0.0293 |
| 8 | 脏污 | `False` | `False` | +6 | +0 | +0 | -0.0607 | -0.0336 |
| 2 | 加强筋打伤 | `False` | `False` | +3 | +0 | +0 | 0.0323 | 0.0248 |
| 12 | 锡膏 | `False` | `False` | +2 | +1 | -1 | 0.0098 | -0.0167 |

## Class 9

- Class 9 was the active and ROI-affected class. ROI applications recorded for class 9: `312`.
- Metric Precision changed `0.7214 -> 0.8376` (`0.1163`).
- Metric Recall changed `0.5238 -> 0.5528` (`0.0290`).
- AP50 changed `0.0502` and AP50-95 changed `0.0228`.
- Matched FP at conf 0.25 changed `50 -> 38` (`-12`).
Class 9 is not the primary Precision-drop class: its metric Precision/AP improved and matched FP decreased at the analysis threshold. The failure is dominated by non-active class FP increases.

## Non-Active Regression

| class | name | dFP | dAP50 | dAP50-95 | active | ROI affected |
|---:|---|---:|---:|---:|---|---|
| 2 | 加强筋打伤 | +3 | 0.0323 | 0.0248 | `False` | `False` |
| 3 | 开裂 | -5 | 0.0000 | -0.0995 | `False` | `False` |
| 4 | 油污 | +1 | -0.0011 | 0.0147 | `False` | `False` |
| 5 | 浅划伤 | +14 | -0.0108 | -0.0620 | `False` | `False` |
| 6 | 漏背锡 | +12 | -0.0259 | -0.0293 | `False` | `False` |
| 8 | 脏污 | +6 | -0.0607 | -0.0336 | `False` | `False` |
| 10 | 锡丝残留 | +0 | -0.0150 | -0.0113 | `False` | `False` |
| 12 | 锡膏 | +2 | 0.0098 | -0.0167 | `False` | `False` |

## Confidence Distribution

- Global FP count at conf 0.25: clean `344`, CP-CATF `354`.
- Global FP confidence mean: clean `0.5899`, CP-CATF `0.5621`.
- Global high-confidence FP count >=0.5: clean `208`, CP-CATF `186`.
The FP count increases while the mean FP confidence slightly decreases, which means the precision risk is mostly broader low-to-mid confidence spillover rather than a simple shift to more high-confidence false positives. Thresholding can suppress part of this tail, but the final-val tuned threshold is leakage-only and cannot be used as a paper result.

## Threshold Calibration

- Target precision: clean seed0 - 0.01 = `0.7413`.
- Final-val diagnostic calibration selected threshold `0.450` and reaches precision `0.7642` / recall `0.7448` on final val. This is leakage analysis only.
- Probe-based calibration selected threshold `0.350` on probe; applied to final val it reaches precision `0.7162` / recall `0.7724`.
- Probe-based threshold restores precision within clean-0.01: `False`.

## Causal Probe Gap

- Paper-mode causal probe accepted class `9` at epoch `25` with causal score `0.01726315789473684`.
- The accept was enough to prove the execution chain, but the final outcome shows the probe underweighted precision risk.
- The likely missing guard is not another dataset-specific blacklist; it is a precision-aware accept condition using the probe split.
- The probe accepted class9 because local benefit looked positive, but it did not sufficiently penalize non-active FP spillover and estimated operating-point precision loss.

## Recommendations

- Add a precision-aware accept gate before continuing performance validation.
- Do not continue seed1/seed2 as paper-mode performance validation yet.
- Do not retrain seed0 immediately without first deciding the precision-risk gate.
