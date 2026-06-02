# Diagnosis-Only In-Loop Control Multiseed Summary

Generated: `2026-06-02T19:04:47`

This experiment ran YOLO default with the in-loop diagnosis callback enabled, but industrial augmentation, ROI augmentation, sample-aware routing, policy mutation, and threshold mutation disabled.

## Integrity

- Seeds: `[0, 1, 2]`
- Epochs: `50` per seed.
- Train images: `2301` per seed.
- Fixed augmented dataset generated: `false`.
- YOLO default augmentation: enabled.
- Industrial augmentation: disabled.
- ROI-aware augmentation: disabled.
- Sample-aware routing: disabled.
- Policy update applied: `0` for every seed.

## Metrics

| seed | clean P/R/mAP50/mAP50-95 | CATF-v2 P/R/mAP50/mAP50-95 | diagnosis-only P/R/mAP50/mAP50-95 | diag delta vs clean | diag delta vs CATF-v2 | diag constraint_failed | CATF-v2 constraint_failed | CATF-v2+threshold constraint_failed |
|---:|---|---|---|---|---|:---:|:---:|:---:|
| 0 | 0.7846/0.6765/0.7347/0.4759 | 0.7429/0.6982/0.7570/0.5023 | 0.7846/0.6765/0.7347/0.4759 | +0.0000/+0.0000/+0.0000/+0.0000 | +0.0417/-0.0218/-0.0223/-0.0264 | false | true | false |
| 1 | 0.7725/0.6477/0.7542/0.4799 | 0.7691/0.6950/0.7549/0.4898 | 0.7725/0.6477/0.7542/0.4799 | +0.0000/+0.0000/+0.0000/+0.0000 | +0.0034/-0.0474/-0.0007/-0.0099 | false | false | false |
| 2 | 0.6962/0.7286/0.7692/0.5224 | 0.8358/0.6039/0.7423/0.4938 | 0.6962/0.7286/0.7692/0.5224 | +0.0000/+0.0000/+0.0000/+0.0000 | -0.1395/+0.1247/+0.0269/+0.0285 | false | true | true |

## Control Counters

| seed | epoch continuous | feedback epochs | industrial samples | ROI applied | policy update applied | bbox/class legal |
|---:|:---:|---|---:|---:|---:|:---:|
| 0 | true | [5, 10, 15, 20, 25, 30, 35, 40, 45] | 0 | 0 | 0 | true |
| 1 | true | [5, 10, 15, 20, 25, 30, 35, 40, 45] | 0 | 0 | 0 | true |
| 2 | true | [5, 10, 15, 20, 25, 30, 35, 40, 45] | 0 | 0 | 0 | true |

## Answers

- Diagnosis-only changed training result: `false`.
- Diagnosis-only constraint pass count: `3/3`.
- CATF-v2 constraint pass count before threshold calibration: `1/3`.
- CATF-v2 constraint pass count after post-hoc threshold calibration: `2/3`.
- Seed 1 CATF-v2 success is not explained by the diagnosis callback alone because diagnosis-only reproduced clean native exactly. It can still reflect CATF-v2 industrial-enabled training-path differences because CATF-v2 seed 1 reported zero actual industrial/ROI augmentation.
- CATF-v2 + post-hoc threshold calibration has independent value as a deployment/post-processing layer: it repairs seed 0 and raises pass count to 2/3, but seed 2 remains unrepaired.
- A method claim should separate three effects: clean YOLO default, diagnosis callback only, and CATF-v2 industrial-enabled path with threshold calibration.

## Report Paths

- seed 0: `outputs\experiments\diagnosis_only_inloop_control_50ep\seed_0\reports\final_report.md`, diagnosis history `outputs\experiments\diagnosis_only_inloop_control_50ep\seed_0\reports\diagnosis_history.json`
- seed 1: `outputs\experiments\diagnosis_only_inloop_control_50ep\seed_1\reports\final_report.md`, diagnosis history `outputs\experiments\diagnosis_only_inloop_control_50ep\seed_1\reports\diagnosis_history.json`
- seed 2: `outputs\experiments\diagnosis_only_inloop_control_50ep\seed_2\reports\final_report.md`, diagnosis history `outputs\experiments\diagnosis_only_inloop_control_50ep\seed_2\reports\diagnosis_history.json`
