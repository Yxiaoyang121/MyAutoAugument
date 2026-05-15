# 推荐实验协议

## 实验组

1. Baseline：原始数据训练 YOLO；
2. Fixed Augment：人工固定增强策略；
3. Random Augment：随机增强策略；
4. Ours：诊断驱动增强策略。

## 消融实验

1. Ours 完整方法；
2. w/o Diagnosis：去掉诊断模块，随机生成策略；
3. w/o Proxy Filter：去掉代理指标过滤；
4. w/o Short Training：去掉短训练筛选；
5. Fixed Strong Augment：固定强增强；
6. Fixed Weak Augment：固定弱增强。

## 评价指标

- mAP50；
- mAP50-95；
- Precision；
- Recall；
- small_object_recall；
- FP；
- FN；
- training_time；
- GPU_memory；
- policy_search_cost。

## 推荐执行顺序

1. 先跑 Baseline，保存 baseline metrics、best.pt、last.pt 和训练日志；
2. 使用 baseline best.pt 对验证集预测，保存 prediction labels；
3. 生成 `diagnosis.json` 和 `diagnosis_summary.md`；
4. 生成 `candidate_policies.json`；
5. 运行 proxy evaluation，输出 `proxy_metrics.json` 和 `proxy_ranking.json`；
6. 对 Top-K 策略进行短周期训练，输出 `short_train_results.json` 和 `selected_policy.json`；
7. 构建最终增强数据集；
8. 训练最终 YOLO；
9. 汇总 `experiment_summary.md`、`experiment_summary.json` 和 `tables_for_paper.md`。

## 结果记录要求

每个实验组和消融实验都应记录：

- 数据集路径和 `data.yaml`；
- 模型权重路径；
- 训练命令；
- stdout / stderr；
- epoch、imgsz、batch、workers、device；
- best.pt / last.pt；
- 验证指标；
- 诊断文件；
- 策略文件；
- 代理评价文件；
- trial 记录。

## 注意事项

本方法比较的是 YOLO 训练框架中的数据增强策略优化，不比较 YOLO 网络结构改动。论文中应明确说明 Backbone、Neck 和 Head 保持不变。
