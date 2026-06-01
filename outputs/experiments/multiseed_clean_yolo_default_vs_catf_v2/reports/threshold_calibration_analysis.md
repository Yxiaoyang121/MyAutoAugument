# CATF-v2 Threshold Calibration Analysis

Generated: `2026-06-01T13:08:26`

| seed | raise threshold classes | lower threshold classes | likely repair effect |
|---:|---|---|---|
| 0 | ['0:OK2', '1:OK3', '3:开裂', '4:油污', '5:浅划伤', '6:漏背锡', '7:碰伤', '8:脏污', '9:轮廓划伤', '10:锡丝残留', '12:锡膏'] | [] | Precision repair candidate |
| 1 | ['0:OK2', '1:OK3', '4:油污', '5:浅划伤', '6:漏背锡', '7:碰伤', '8:脏污', '9:轮廓划伤', '10:锡丝残留', '12:锡膏'] | ['11:锡尖'] | Already passes constraints |
| 2 | ['0:OK2', '1:OK3', '3:开裂', '4:油污', '5:浅划伤', '6:漏背锡', '7:碰伤', '8:脏污', '9:轮廓划伤', '10:锡丝残留', '12:锡膏'] | ['2:加强筋打伤', '11:锡尖'] | Recall repair partial |

## Conclusions

- Seed 0 Precision loss is the clearest threshold-calibration target; many high-FP classes are recommended for threshold raise.
- Seed 2 Recall loss cannot be solved cleanly by threshold alone because many classes are simultaneously high-FP guarded and recommended for threshold raise.
- A defensible next protocol is `YOLO default + conservative CATF-v2 training` followed by `analysis-only per-class threshold calibration`, then a separate post-processing validation.
