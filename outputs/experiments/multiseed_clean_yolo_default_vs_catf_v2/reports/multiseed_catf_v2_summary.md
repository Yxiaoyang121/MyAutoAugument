# Multiseed Clean Native YOLO Default vs CATF-v2 Feedback

Generated: `2026-06-01T08:54:50`

## Setup

- Seeds: `0, 1, 2`. Seed42 remains a positive single-seed result and is not included in the multiseed mean.
- Clean group: pure native Ultralytics YOLO default augmentation, reused from existing clean native runs.
- CATF-v2 group: single-run continuous in-loop feedback with official YOLO default augmentation kept enabled; class-aware, issue-aware, sample-aware, ROI-aware industrial augmentation enabled.
- No self-implemented mosaic/randaugment replacement; copy_paste remains `pending_object_bank_design`.
- Train images remain `2301`: `true`; fixed augmented dataset generated: `false`.
- Single-run and continuous epochs across CATF-v2 runs: `true`.
- BBox/class legality across CATF-v2 runs: `true`.

## Per-Seed Results

| Seed | Clean P | Clean R | Clean mAP50 | Clean mAP50-95 | CATF-v2 P | CATF-v2 R | CATF-v2 mAP50 | CATF-v2 mAP50-95 | ?P | ?R | ?mAP50 | ?mAP50-95 | Constraint failed | Verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|
| 0 | 0.7846 | 0.6765 | 0.7347 | 0.4759 | 0.7429 | 0.6982 | 0.7570 | 0.5023 | -0.0417 | +0.0218 | +0.0223 | +0.0264 | true | fail: precision_drop_gt_0.01 |
| 1 | 0.7725 | 0.6477 | 0.7542 | 0.4799 | 0.7691 | 0.6950 | 0.7549 | 0.4898 | -0.0034 | +0.0474 | +0.0007 | +0.0099 | false | pass: Recall/mAP improve within constraints |
| 2 | 0.6962 | 0.7286 | 0.7692 | 0.5224 | 0.8358 | 0.6039 | 0.7423 | 0.4938 | +0.1395 | -0.1247 | -0.0269 | -0.0285 | true | fail: map50_drop_gt_0.01, map50_95_drop_gt_0.01 |

## Mean ? Std

`std` is sample standard deviation across seeds 0/1/2.

| Group | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Clean native | 0.7511 ? 0.0479 | 0.6843 ? 0.0410 | 0.7527 ? 0.0173 | 0.4927 ? 0.0258 |
| CATF-v2 | 0.7826 ? 0.0479 | 0.6657 ? 0.0536 | 0.7514 ? 0.0080 | 0.4953 ? 0.0063 |
| ? CATF-v2 - clean | +0.0315 ? 0.0955 | -0.0185 ? 0.0929 | -0.0013 ? 0.0247 | +0.0026 ? 0.0282 |

## Constraint And Win Counts

- CATF-v2 better under industrial constraints: `1/3`.
- Constraint failed seeds: `2/3`.
- All-metric-win seeds: `0/3`.
- Metric win counts: Precision `1/3`, Recall `2/3`, mAP50 `2/3`, mAP50-95 `2/3`.

## Activation And ROI Audit

- OK2 ever active: `false`.
- OK3 ever active: `false`.
- OK3 ROI applied total: `0`.
- Active class counts from policy matrices: `{'8:脏污': 1, '11:锡尖': 1}`.
- ROI affected class totals: `{'8:脏污': 3, '11:锡尖': 20}`.

- Seed 0: OK2 active=`false`, OK3 active=`false`, OK3 ROI=`0`, ROI total=`20`, active_classes=`{'11:锡尖': 1}`, ROI classes=`{'11': 20}`.
- Seed 1: OK2 active=`false`, OK3 active=`false`, OK3 ROI=`0`, ROI total=`0`, active_classes=`{}`, ROI classes=`{}`.
- Seed 2: OK2 active=`false`, OK3 active=`false`, OK3 ROI=`0`, ROI total=`3`, active_classes=`{'8:脏污': 1}`, ROI classes=`{'8': 3}`.

## Feedback Control Statistics

- Policy actions total: `{'shrink': 14, 'accept': 2, 'observe': 5, 'freeze': 6}`.
- Class actions total: `{'propose': 2, 'observe': 8}`.
- Rollback/cooldown/freeze event totals: `0/0/12`.
- Frozen class records across freeze epochs: `78`.

- Seed 0: actions=`{'shrink': 2, 'accept': 1, 'observe': 4, 'freeze': 2}`, class_actions=`{'propose': 1}`, rollback/cooldown/freeze=`0/0/4`, max_active_classes_per_feedback=`1`.
- Seed 1: actions=`{'shrink': 7, 'freeze': 2}`, class_actions=`{'observe': 4}`, rollback/cooldown/freeze=`0/0/4`, max_active_classes_per_feedback=`0`.
- Seed 2: actions=`{'shrink': 5, 'accept': 1, 'observe': 1, 'freeze': 2}`, class_actions=`{'observe': 4, 'propose': 1}`, rollback/cooldown/freeze=`0/0/4`, max_active_classes_per_feedback=`1`.

## CATF-v2 vs CATF-v1

- CATF-v1 constraint failed count: `3/3`; CATF-v2 constraint failed count: `2/3`.
- Assessment: CATF-v2 improved activation control and reduces constraint failures versus CATF-v1, but remains unstable because 2/3 seeds still fail industrial constraints.

## Recommendation

- Recommendation: `not_paper_main_method`.
- CATF-v2 should not yet be claimed as the paper main method on seeds 0/1/2. It is better than CATF-v1 in activation discipline and constraint failure count, and seed1 passes constraints, but seed0 loses too much Precision and seed2 loses Recall/mAP50/mAP50-95. Treat as a promising ablation/controller variant and refine before main-result use.

## Key Paths

- Summary JSON: `outputs\experiments\multiseed_clean_yolo_default_vs_catf_v2\reports\multiseed_catf_v2_summary.json`
- Seed 0 policy history: `outputs\experiments\multiseed_clean_yolo_default_vs_catf_v2\seed_0\catf_v2\reports\policy_history.json`
- Seed 0 best checkpoint: `outputs\experiments\multiseed_clean_yolo_default_vs_catf_v2\seed_0\catf_v2\train\weights\best.pt`
- Seed 1 policy history: `outputs\experiments\multiseed_clean_yolo_default_vs_catf_v2\seed_1\catf_v2\reports\policy_history.json`
- Seed 1 best checkpoint: `outputs\experiments\multiseed_clean_yolo_default_vs_catf_v2\seed_1\catf_v2\train\weights\best.pt`
- Seed 2 policy history: `outputs\experiments\multiseed_clean_yolo_default_vs_catf_v2\seed_2\catf_v2\reports\policy_history.json`
- Seed 2 best checkpoint: `outputs\experiments\multiseed_clean_yolo_default_vs_catf_v2\seed_2\catf_v2\train\weights\best.pt`
