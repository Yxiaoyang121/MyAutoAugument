# CATF-v2 Recall Drop Analysis

Generated: `2026-06-01T13:08:26`

- Seed 2 global delta: `{'precision': 0.1395, 'recall': -0.1247, 'map50': -0.0269, 'map50_95': -0.0285}`.
- Top approximate FN increases:

| class | ΔFN | ΔRecall | ΔAP50 | ΔAP50-95 | active? | threshold recommendation |
|---|---:|---:|---:|---:|:---:|---|
| 7:碰伤 | +28.5093 | -0.1060 | +0.0388 | +0.0308 | false | 0.25 -> 0.4 high_fp_raise_threshold |
| 6:漏背锡 | +7.0000 | -0.1522 | -0.0424 | +0.0163 | false | 0.25 -> 0.4 high_fp_raise_threshold |
| 9:轮廓划伤 | +7.0000 | -0.0833 | +0.0064 | -0.0263 | false | 0.25 -> 0.4 high_fp_raise_threshold |
| 8:脏污 | +6.0446 | -0.1439 | -0.0420 | -0.0358 | true | 0.25 -> 0.4 high_fp_raise_threshold |
| 2:加强筋打伤 | +4.0000 | -0.5000 | -0.2966 | -0.2312 | false | 0.25 -> 0.15 low_recall_lower_threshold |
| 5:浅划伤 | +3.7222 | -0.2863 | +0.0101 | +0.0438 | false | 0.25 -> 0.4 high_fp_raise_threshold |
| 12:锡膏 | +2.8849 | -0.0627 | -0.0737 | -0.0443 | false | 0.25 -> 0.4 high_fp_raise_threshold |
| 10:锡丝残留 | +2.1390 | -0.1783 | +0.0336 | -0.0059 | false | 0.25 -> 0.4 high_fp_raise_threshold |

## Answers

- Seed 2 Recall loss is broad and not concentrated in the active 脏污 class.
- The P/R pattern strongly suggests conservative confidence/detection behavior: Precision rises by +0.1395 while Recall drops by -0.1247.
- Clean seed2 was already a high-Recall trajectory; CATF-v2 should likely reduce or disable intervention when the reference curve already outperforms on Recall/mAP.
- Shrink/freeze behavior leaves too little opportunity to rescue classes after early negative drift.
- Seed 1 had zero industrial augmentation but still changed metrics, so CATF-v2 evaluation should include an in-loop diagnosis-only control to isolate callback/RNG effects.
