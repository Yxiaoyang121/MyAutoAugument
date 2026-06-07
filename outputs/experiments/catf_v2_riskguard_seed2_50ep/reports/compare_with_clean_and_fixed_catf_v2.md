# CATF-v2 RiskGuard Seed2 Validation

## Verdict

- Completed 50ep: `true`
- constraint_failed: `true`
- Failure reasons: `['precision_drop_gt_0.01', 'map50_drop_gt_0.01', 'map50_95_drop_gt_0.01']`
- Seed0/seed1 sanity check: `not run` because seed2 did not pass the constraint gate.
- Interpretation: RiskGuard successfully blocks the audited class 9 texture intervention, but it is not sufficient to recover seed2 overall.

## Metrics

| group | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| clean seed2 | 0.6962 | 0.7286 | 0.7692 | 0.5224 |
| fixed CATF-v2 seed2 | 0.7637 | 0.6863 | 0.7582 | 0.4967 |
| RiskGuard seed2 | 0.6394 | 0.7231 | 0.7552 | 0.5073 |

## Deltas

- RiskGuard vs clean: P `-0.0569`, R `-0.0056`, mAP50 `-0.0140`, mAP50-95 `-0.0150`.
- RiskGuard vs fixed CATF-v2: P `-0.1243`, R `0.0367`, mAP50 `-0.0030`, mAP50-95 `0.0107`.

## RiskGuard Events

- Epoch5 class 9 dominant issue: `texture_boundary_weak`
- Class 9 texture ops blocked: `['local_contrast', 'sharpen_mild']`
- Blocked op count: `2`
- Sampler-only fallback: `true`
- Sample weighting effective: `false`; dataloader integration remains pending.
- Event JSON: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\catf_v2_riskguard_seed2_50ep\reports\riskguard_events.json`

## Augmentation Stats

- Industrial samples augmented: `13` vs fixed `86`.
- ROI applied: `15` vs fixed `90`.
- ROI affected classes: `{'12': 15}`.
- Router random draw count: `960`.
- OK3 active: `false`; OK3 ROI applied: `0`.

## Class 9

- Recall clean/fixed/RiskGuard: `0.6310` / `0.4643` / `0.7960`.
- AP50 clean/fixed/RiskGuard: `0.7440` / `0.6973` / `0.7944`.
- AP50-95 clean/fixed/RiskGuard: `0.4139` / `0.3484` / `0.4198`.
- Class 9 recovered versus fixed CATF-v2 after blocking the texture ROI combination, but global constraints still failed.

## Per-Class Residual Risk

| class | name | dRecall vs clean | dAP50 vs clean | dAP50-95 vs clean |
|---:|---|---:|---:|---:|
| 3 | 开裂 | 0.0000 | 0.0000 | -0.0995 |
| 2 | 加强筋打伤 | -0.1955 | -0.0364 | -0.0818 |
| 8 | 脏污 | -0.1052 | -0.0836 | -0.0433 |
| 12 | 锡膏 | 0.0429 | -0.0072 | -0.0199 |
| 4 | 油污 | 0.0124 | -0.0242 | -0.0170 |
| 11 | 锡尖 | 0.0417 | 0.0018 | -0.0146 |
| 7 | 碰伤 | 0.0029 | -0.0119 | -0.0082 |
| 5 | 浅划伤 | 0.0000 | -0.0425 | -0.0025 |
| 10 | 锡丝残留 | 0.0000 | -0.0355 | 0.0666 |

## Conclusion

- RiskGuard should be added as a targeted CATF-v2 safety mechanism because it prevents the known class 9 + ROI texture failure path without seed-specific logic.
- This seed2 validation does not justify running seed0/seed1 sanity yet: seed2 still fails due to residual global Precision/mAP degradation and class12-only ROI activity.
- The next minimal fix should combine RiskGuard with a gate/causal probe that rejects later class-op interventions when global Precision/mAP constraints deteriorate.
