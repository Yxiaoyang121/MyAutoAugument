# 诊断驱动数据增强框架

## 1. 研究动机

工业缺陷检测通常存在小目标、低对比、类别不均衡、背景干扰强和标注样本有限等问题。直接对增强空间进行大量随机搜索，需要反复训练 YOLO 检测器，计算成本高，且难以解释某个增强策略为何有效。

本项目的新定位是：基于验证误差诊断驱动的数据增强策略优化框架，用于工业缺陷检测任务。核心思想是先观察 baseline YOLO 在验证集上的错误，再根据错误类型生成有针对性的增强策略。

本文方法不是修改 YOLO 网络结构，而是提出一种面向检测模型训练阶段的诊断驱动数据增强优化框架。

## 2. 为什么不采用大量随机搜索

大量随机搜索的主要问题包括：

- 每个策略都需要训练和验证，实验成本高；
- 搜索过程对错误类型不敏感，无法解释增强策略来源；
- 强增强策略可能破坏 bbox 或产生不合理曝光；
- 随机策略很容易把计算资源浪费在明显不适合工业缺陷图像的组合上。

因此，本框架将策略生成从“无目标随机采样”改为“验证误差感知生成”，并在训练前加入代理指标过滤。

## 3. 整体框架

完整流程包括：

1. 使用原始训练集训练 baseline YOLO；
2. 使用 baseline 模型在验证集上预测；
3. 对 TP、FP、FN、定位偏差、小目标召回、低对比漏检和类别级指标进行诊断；
4. 将诊断问题映射为候选增强策略；
5. 使用 bbox 保留率、bbox 有效率、曝光合理性、增强多样性和类别分布变化等代理指标过滤策略；
6. 对少量 Top-K 候选策略进行短周期训练验证；
7. 选择最优增强策略；
8. 构建最终增强训练集；
9. 训练最终 YOLO 检测模型；
10. 输出可审计实验报告。

## 4. Baseline YOLO 检测器

Baseline 模块只使用原始训练集训练 YOLO。它记录训练命令、`data.yaml`、数据集路径、epoch、imgsz、batch、workers、stdout、stderr、best.pt、last.pt 和验证指标。

该模块的作用是提供错误诊断的参照模型，而不是作为最终方法的网络结构创新。

## 5. 验证误差诊断模块

诊断模块读取验证集真实标签和 baseline 预测结果，统计：

- TP、FP、FN；
- 全局 Precision / Recall；
- 类别级 Precision / Recall；
- 每类 FN / FP 分布；
- 小目标召回不足；
- 低对比缺陷漏检；
- 定位偏差；
- 类别不均衡；
- 每张图像错误摘要。

输出包括 `diagnosis.json` 和 `diagnosis_summary.md`。`diagnosis.json` 是后续策略生成的主输入。

## 6. 诊断结果到增强策略的映射规则

映射规则不是随机采样，而是由错误类型触发：

- 小目标召回不足：推荐 tiling、object-aware crop、scale、copy-paste、小幅几何增强；
- 低对比缺陷漏检：推荐 contrast、gamma、CLAHE、brightness、sharpen；
- 定位偏差较大：降低 rotate、shear、perspective、crop 的强度，优先 mild scale / translate；
- FP 较多：降低强噪声、强模糊、过强光照扰动，保留轻量对比度调整；
- 类别不均衡：使用 class-aware sampling 或 targeted augmentation；
- 背景干扰强：增加背景多样性、轻量噪声和光照扰动。

当前代码只把已实现且 bbox 同步安全的算子写入可执行 policy；tiling、copy-paste、class-aware sampling 作为数据级建议记录在解释和元数据中。

## 7. 代理指标筛选

代理评价模块在不训练 YOLO 的情况下先过滤明显不合理的策略。指标包括：

- bbox_retention；
- bbox_valid_rate；
- diversity_score；
- exposure_score；
- mean_intensity；
- mean_std；
- class_distribution_change；
- small_object_retention；
- invalid_sample_count。

过滤规则重点关注 bbox 保留率、bbox 有效率、曝光异常和小目标丢失。代理评价不能证明策略一定最优，但能避免把训练成本浪费在明显破坏样本的策略上。

## 8. 短周期训练选择

短训练模块只对代理评价后的 Top-K 策略进行少量 epoch 验证，默认 5 或 10 epoch，`workers=0`。每个 trial 保存：

- `policy.json`；
- `proxy_metrics.json`；
- `train_command.txt`；
- `train_stdout.log`；
- `train_stderr.log`；
- `val_metrics.json`；
- `trial_record.json`。

选择依据包括 mAP50、mAP50-95、Recall、small_object_recall、FP 和 FN。

## 9. 最终增强训练

最终数据集构建模块保留原始图像，并用 `selected_policy.json` 生成增强图像，输出标准 YOLO 格式和新的 `data.yaml`。最终训练模块使用该增强训练集训练 YOLO，并输出 `final_train_metrics.json`、`final_val_metrics.json` 和 `final_report.md`。

## 10. 消融实验设计

推荐消融：

- Ours 完整方法；
- w/o Diagnosis：去掉诊断模块，随机生成策略；
- w/o Proxy Filter：去掉代理指标过滤；
- w/o Short Training：去掉短训练筛选；
- Fixed Strong Augment：固定强增强；
- Fixed Weak Augment：固定弱增强。

评价指标包括 mAP50、mAP50-95、Precision、Recall、small_object_recall、FP、FN、training_time、GPU_memory 和 policy_search_cost。

## 11. 论文中推荐的表述方式

推荐表述：

- 诊断驱动数据增强框架；
- 验证误差感知增强策略优化方法；
- 工业缺陷检测自适应增强训练框架；
- 基于验证误差诊断的 YOLO 训练阶段增强优化框架。

避免表述：

- YOLO 网络结构改进；
- 改进 YOLO Backbone / Neck / Head；
- 新检测头或新特征融合结构。

## 论文贡献点建议

1. 提出一种面向工业缺陷检测的验证误差感知数据增强框架，可根据漏检、误检、小目标召回不足、低对比缺陷等错误类型自适应生成增强策略。

2. 设计结合 bbox 有效性、目标保留率、曝光分布、增强多样性和类别分布变化的代理评价指标，用于在无需完整训练的情况下过滤低质量增强策略。

3. 构建代理评价与短周期训练相结合的低成本策略选择流程，在降低搜索成本的同时提升工业缺陷检测模型的召回率和检测精度。
