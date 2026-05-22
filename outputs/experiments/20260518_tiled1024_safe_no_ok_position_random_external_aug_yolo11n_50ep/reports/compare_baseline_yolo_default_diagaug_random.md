# Baseline vs YOLO Default vs DiagAug vs Random External

| run | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| No-YOLO-Aug baseline | 0.690 | 0.615 | 0.669 | 0.434 |
| YOLO default aug | 0.785 | 0.676 | 0.735 | 0.476 |
| DiagAug | 0.686 | 0.688 | 0.717 | 0.496 |
| Random external aug | 0.750 | 0.668 | 0.734 | 0.501 |

## Overall Delta For Random External

| reference | delta Precision | delta Recall | delta mAP50 | delta mAP50-95 |
|---|---:|---:|---:|---:|
| baseline | +0.060 | +0.053 | +0.065 | +0.067 |
| YOLO default | -0.035 | -0.008 | -0.001 | +0.025 |
| DiagAug | +0.064 | -0.020 | +0.017 | +0.005 |

## Per-Class Recall/AP50

| class | baseline R/AP50 | YOLO default R/AP50 | DiagAug R/AP50 | random R/AP50 |
|---|---:|---:|---:|---:|
| OK2 | 1.000/0.995 | 0.984/0.993 | 1.000/0.993 | 1.000/0.995 |
| OK3 | 0.978/0.970 | 0.967/0.994 | 0.989/0.973 | 0.995/0.967 |
| 加强筋打伤 | 0.875/0.955 | 0.875/0.949 | 0.625/0.747 | 0.875/0.982 |
| 开裂 | 0.000/0.111 | 1.000/0.663 | 1.000/0.995 | 1.000/0.995 |
| 油污 | 0.333/0.242 | 0.278/0.292 | 0.389/0.264 | 0.389/0.290 |
| 浅划伤 | 0.550/0.712 | 0.462/0.551 | 0.615/0.581 | 0.506/0.622 |
| 漏背锡 | 0.712/0.694 | 0.500/0.686 | 0.561/0.508 | 0.587/0.608 |
| 碰伤 | 0.617/0.617 | 0.565/0.723 | 0.546/0.640 | 0.513/0.638 |
| 脏污 | 0.381/0.440 | 0.500/0.517 | 0.476/0.443 | 0.381/0.446 |
| 轮廓划伤 | 0.511/0.717 | 0.571/0.747 | 0.563/0.666 | 0.607/0.709 |
| 锡丝残留 | 0.583/0.619 | 0.650/0.776 | 0.574/0.774 | 0.366/0.660 |
| 锡尖 | 0.852/0.906 | 0.773/0.951 | 0.888/0.974 | 0.896/0.929 |
| 锡膏 | 0.609/0.725 | 0.668/0.709 | 0.717/0.760 | 0.567/0.698 |

## Interpretation Notes

- Random external augmentation uses the same offline expansion size as DiagAug: original train images plus one augmented copy per train image.
- YOLO built-in augmentation is disabled for both DiagAug final training and this random external control.
- Class-aware policy evaluation remains a 5 epoch short-training result only; it is not a formal 50 epoch comparator here.
