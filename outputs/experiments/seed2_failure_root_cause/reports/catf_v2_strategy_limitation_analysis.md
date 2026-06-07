# CATF-v2 策略不足分析与后续改进建议

本报告基于已有 seed2 failure root-cause audit 结果整理。未训练新模型，未运行消融，未修改 CATF-v2 规则。

## 一、Seed2 失败链条总结

### 时间线

| 阶段 | 证据 | 结论 |
|---|---|---|
| clean seed2 | P=0.6962, R=0.7286, mAP50=0.7692, mAP50-95=0.5224 | clean baseline 本身较强，尤其 Recall 和 mAP50-95 是当前 seed2 保护目标 |
| fixed CATF-v2 seed2 | P=0.7637, R=0.6863, mAP50=0.7582, mAP50-95=0.4967; dP=+0.0674, dR=-0.0423, dM50=-0.0110, dM95=-0.0257 | 不是 Precision 问题，而是更保守的漏检和定位质量下降 |
| epoch 5 | epoch 5 accept/propose class 9 texture_boundary_weak with sharpen_mild and local_contrast | 最可疑的首次策略更新，发生在退化前 |
| epoch 6-8 | Recall first lag=6; mAP50 clear lag=8; mAP50-95 clear lag=8 | 退化紧跟 epoch5 候选增强窗口出现 |
| final per-class | class 8 (dR=+0.0301, dAP50=-0.0559, dAP50-95=-0.0502, active=true, ROI=6), class 9 (dR=-0.1667, dAP50=-0.0466, dAP50-95=-0.0654, active=true, ROI=25) | active class 与退化类存在重叠，class 9 最关键 |
| non-active classes | class 5 (dR=-0.2704, dAP50=+0.1072, dAP50-95=+0.1010, active=false, ROI=0), class 3 (dR=+0.0000, dAP50=+0.0000, dAP50-95=-0.1492, active=false, ROI=0), class 2 (dR=-0.1250, dAP50=+0.0045, dAP50-95=-0.0463, active=false, ROI=0), class 10 (dR=+0.0000, dAP50=-0.0748, dAP50-95=-0.0880, active=false, ROI=0), class 6 (dR=-0.0745, dAP50=-0.0370, dAP50-95=+0.0196, active=false, ROI=0) | 存在增强 A 类、伤害 B 类的非目标类退化 |

### 增强执行证据

- Active class 最可疑：`class 9`。
- 最可疑算子组合：`ROI texture ops as a pair: sharpen_mild + local_contrast; current logs cannot isolate one op`。
- Industrial samples augmented：`86`。
- ROI applied：`90`。
- Router random draw count：`5610`。
- ROI affected classes：`{'8': 6, '9': 25, '11': 59}`。
- Prediction diff：clean 检出但 fixed 漏检 `24` 个 GT；clean 高 IoU 但 fixed 定位变差 `11` 个 GT。

## 二、当前 CATF-v2 策略的主要不足

1. **诊断结果不能直接等价于增强收益**
   - 证据：epoch5 将 class 9 诊断为 texture_boundary_weak 且置信度高，但随后 Recall 在 epoch6 落后 clean，mAP50/mAP50-95 在 epoch8 明显落后。
2. **active class 的局部问题并不保证该类受益**
   - 证据：class 9 是 epoch5 active class，ROI applied=25，但最终 Recall -0.1667、AP50 -0.0466、AP50-95 -0.0654，估算 FN +14。
3. **缺少增强前因果验证**
   - 证据：当前策略在诊断后直接进入图像增强，未先验证候选增强是否能恢复 FN、是否增加 FP、是否伤害非 active 类。
4. **只约束 active class 不足以控制 non-active regression**
   - 证据：class 10、12、6、5、3、2 等非 ROI 类出现 AP 或 Recall 退化；说明增强 A 类可能改变共享特征并伤害 B 类。
5. **算子级风险没有被隔离**
   - 证据：seed2 中 sharpen_mild 与 local_contrast 共同启用，local_contrast applied=44、sharpen_mild applied=42，现有日志无法拆分单个算子的责任。
6. **fallback/gate 发现问题时权重可能已经被影响**
   - 证据：Gated 在 epoch10 fallback，但 epoch6-10 已经经过候选增强训练；因此 strict no-op 后续路径不等价于 clean 权重轨迹。
7. **强 clean baseline 下策略应更保守**
   - 证据：Safe 与 adaptive-RB strict no-op 完全复现 clean seed2；fixed/Gated 失败，说明 seed2 这类强 clean baseline 更适合 no-op 或 sampler-only。

## 三、这不是“方法失败”

CATF-v2 并非无效。fixed CATF-v2 在 seed0/seed1 上保留了有效收益，seed2 暴露的是策略选择与风险控制不足，而不是类别感知反馈增强方向错误。

| seed | dP fixed-clean | dR fixed-clean | dM50 fixed-clean | dM95 fixed-clean | constraint pass |
|---:|---:|---:|---:|---:|:---:|
| 0 | -0.0060 | -0.0068 | +0.0090 | +0.0136 | true |
| 1 | +0.0127 | +0.0528 | +0.0284 | +0.0390 | true |
| avg 0/1/2 | +0.0247 | +0.0012 | +0.0088 | +0.0090 | n/a |

## 四、论文可用表述：反馈增强策略的局限性分析

反馈增强策略在工业缺陷检测中需要同时考虑收益与风险。我们的实验表明，类别感知反馈增强并非在所有随机种子上都产生一致收益：在 seed0 和 seed1 上，CATF-v2 能带来 mAP 或多指标提升；但在 seed2 上，候选增强使模型呈现更保守的预测行为，即 Precision 上升而 Recall 与定位相关 AP 下降。进一步分析发现，诊断出的类别问题并不能直接等价为图像增强收益。以 class 9 为例，该类被诊断为边界纹理相关问题，但 ROI texture 增强后该类 Recall 和 AP 反而下降，并伴随若干非目标类别退化。这说明反馈增强策略需要在训练前引入因果验证，检查候选增强是否确实恢复漏检、是否引入新的误检，以及是否损害非 active 类。对于工业应用，增强策略不应只最大化 active class 的局部收益，还应将全局 Recall、定位质量和非目标类稳定性纳入风险约束。

## 五、后续最小改进方向

1. **增强前 causal probe**
   - 做法：不更新权重，先验证候选增强是否恢复 FN、是否增加 FP、是否伤害非 active 类。
   - 最小性：只增加干预前验证层，不改变现有 CATF-v2 增强算子或训练流程。
2. **active / non-active 双重约束**
   - 做法：候选策略不仅看 active class 是否改善，还要监控其他类别 AP/Recall 是否退化。
   - 最小性：把已有 per-class diagnosis 与 prediction diff 转成 gate 条件，避免增强 A 类伤害 B 类。
3. **高风险 class-op 组合黑名单**
   - 做法：class 9 + ROI texture 这类已暴露风险的组合进入高风险候选池，必须通过 causal probe 才允许训练。
   - 最小性：不是永久禁用算子，而是对有证据的风险组合提高准入门槛。

## 六、最终结论

1. 是否需要大量消融才能证明当前不足：`false`。已有 seed2 audit 已足够证明策略选择存在不足；大量消融是后续优化，不是证明不足的前提。
2. 当前证据是否足够说明 CATF-v2 策略选择存在不足：`true`。
3. seed2 最主要暴露的问题：诊断到增强的策略选择缺少因果验证，并且 fallback/gate 在权重已受影响后才触发。
4. 下一步最值得实现：`CP-CATF causal probe`。
5. 如何转化为论文创新点：将 seed2 暴露的问题转化为从 diagnosis-driven augmentation 到 causality-checked feedback augmentation 的动机：增强动作必须先通过 FN/FP/non-active regression 风险验证。
