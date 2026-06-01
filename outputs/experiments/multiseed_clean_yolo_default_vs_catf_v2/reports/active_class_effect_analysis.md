# CATF-v2 Active Class Effect Analysis

Generated: `2026-06-01T13:08:26`

| seed | epoch | class | issue | confidence | action | ROI count | ΔRecall | ΔAP50 | ΔAP50-95 | ΔFP | contribution |
|---:|---:|---|---|---:|---|---:|---:|---:|---:|---:|---|
| 0 | 10 | 11:锡尖 | texture_boundary_weak | 0.7700 | accept | 20 | +0.1526 | +0.0236 | +0.0783 | +3.6723 | False |
| 2 | 25 | 8:脏污 | texture_boundary_weak | 0.9200 | accept | 3 | -0.1439 | -0.0420 | -0.0358 | -1.9990 | False |

## Findings

- 锡尖在 seed 0 被激活后本类 Recall/AP 有提升，但全局 Precision 失败，说明收益不是无代价的全局收益。
- 脏污在 seed 2 被激活后本类 Recall/AP 下降且 FP 下降；它符合 seed 2 的高 Precision、低 Recall 保守化模式。
- seed 42 的漏背锡/锡膏激活没有在 seeds 0/1/2 稳定复现，说明诊断选择对 seed 轨迹敏感。
- active class 太少：三次 50 epoch 只有 2 次 class propose，seed 1 完全没有 ROI applied；seed 1 通过约束不能证明 ROI augmentation 有效。
- 存在增强目标与最终退化类不一致的问题；需要 negative effect attribution 来回滚对应类或扩大诊断到实际退化类。
