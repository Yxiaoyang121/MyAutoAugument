# Baseline vs DiagAug vs Random External vs YOLO Default vs search_policy_017

| run | Precision | Recall | mAP50 | mAP50-95 | balanced |
|---|---:|---:|---:|---:|---:|
| No-YOLO-Aug baseline | 0.690 | 0.615 | 0.669 | 0.434 | 0.589200 |
| diag_policy_001 DiagAug | 0.686 | 0.688 | 0.717 | 0.496 | 0.637250 |
| YOLO default aug | 0.785 | 0.676 | 0.735 | 0.476 | 0.652550 |
| Random external aug | 0.750 | 0.668 | 0.734 | 0.501 | 0.650800 |
| search_policy_017 | 0.710 | 0.616 | 0.681 | 0.474 | 0.608450 |

## Overall Delta For search_policy_017

| reference | delta Precision | delta Recall | delta mAP50 | delta mAP50-95 | delta balanced |
|---|---:|---:|---:|---:|---:|
| baseline | +0.020 | +0.001 | +0.012 | +0.040 | +0.019250 |
| diag_policy_001 DiagAug | +0.024 | -0.072 | -0.036 | -0.022 | -0.028800 |
| YOLO default | -0.075 | -0.060 | -0.054 | -0.002 | -0.044100 |
| random external | -0.040 | -0.052 | -0.053 | -0.027 | -0.042350 |

## Per-Class Recall/AP50

| class | baseline R/AP50 | DiagAug R/AP50 | YOLO default R/AP50 | random R/AP50 | search017 R/AP50 |
|---|---:|---:|---:|---:|---:|
| OK2 | 1.000/0.995 | 1.000/0.993 | 0.984/0.993 | 1.000/0.995 | 1.000/0.995 |
| OK3 | 0.978/0.970 | 0.989/0.973 | 0.967/0.994 | 0.995/0.967 | 0.993/0.974 |
| 加强筋打伤 | 0.875/0.955 | 0.625/0.747 | 0.875/0.949 | 0.875/0.982 | 0.375/0.492 |
| 开裂 | 0.000/0.111 | 1.000/0.995 | 1.000/0.663 | 1.000/0.995 | 1.000/0.995 |
| 油污 | 0.333/0.242 | 0.389/0.264 | 0.278/0.292 | 0.389/0.290 | 0.429/0.394 |
| 浅划伤 | 0.550/0.712 | 0.615/0.581 | 0.462/0.551 | 0.506/0.622 | 0.615/0.643 |
| 漏背锡 | 0.712/0.694 | 0.561/0.508 | 0.500/0.686 | 0.587/0.608 | 0.565/0.615 |
| 碰伤 | 0.617/0.617 | 0.546/0.640 | 0.565/0.723 | 0.513/0.638 | 0.480/0.596 |
| 脏污 | 0.381/0.440 | 0.476/0.443 | 0.500/0.517 | 0.381/0.446 | 0.381/0.488 |
| 轮廓划伤 | 0.511/0.717 | 0.563/0.666 | 0.571/0.747 | 0.607/0.709 | 0.400/0.669 |
| 锡丝残留 | 0.583/0.619 | 0.574/0.774 | 0.650/0.776 | 0.366/0.660 | 0.333/0.329 |
| 锡尖 | 0.852/0.906 | 0.888/0.974 | 0.773/0.951 | 0.896/0.929 | 0.852/0.910 |
| 锡膏 | 0.609/0.725 | 0.717/0.760 | 0.668/0.709 | 0.567/0.698 | 0.587/0.749 |

## Formal Ranking

- Best by mAP50-95: `random_external`
- Best by balanced score: `yolo_default`
- search_policy_017 exceeds random external by mAP50-95: `false`
- search_policy_017 exceeds diag_policy_001 by mAP50-95: `false`

## Notes

- All formal external/offline augmentation runs use original train images plus one augmented copy per train image.
- YOLO built-in augmentation is disabled for baseline, DiagAug, random external, and search_policy_017 unless explicitly listed as the YOLO default control.
- `search_policy_017` is compared as a formal 50 epoch result, not as the previous 5 epoch short-training trend check.
