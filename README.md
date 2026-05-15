# Diagnostic-Driven AutoAugment for Industrial Defect Detection

本项目是一个面向工业缺陷目标检测的 **验证误差诊断驱动数据增强策略优化框架**。当前研究主线不再是“大量随机搜索增强策略并反复完整训练 YOLO”，而是：

1. 训练 baseline YOLO 检测器；
2. 在验证集上预测并分析 TP / FP / FN、定位偏差、小目标召回、低对比漏检和类别不均衡；
3. 根据诊断结果生成候选增强策略；
4. 使用代理指标过滤明显不合理策略；
5. 对少量 Top-K 策略进行短周期训练验证；
6. 选择最优策略并构建最终增强训练集；
7. 训练最终 YOLO 检测模型；
8. 输出可审计实验报告。

本项目 **不是 YOLO Backbone / Neck / Head 网络结构改进**。论文中建议表述为：

- 诊断驱动数据增强框架；
- 验证误差感知增强策略优化方法；
- 工业缺陷检测自适应增强训练框架。

## 已有能力

- YOLO 格式数据集读取与写出；
- 检测样本统一结构：`image + bboxes + labels`；
- bbox 同步变换、裁剪和合法性过滤；
- 颜色、模糊、噪声、几何类增强算子；
- policy 定义、应用和随机搜索；
- proxy 指标评价；
- YOLO train / val / predict 命令封装；
- 验证集错误诊断和增强建议；
- trial 记录和审计输出。

## 新增诊断驱动流水线

统一入口：

```bash
python scripts/run_diagnostic_augmentation_pipeline.py ^
  --dataset-root E:\TJGY\DataSet2_fixed ^
  --data-yaml E:\TJGY\DataSet2_fixed\data.yaml ^
  --output-dir outputs\diagnostic_aug_pipeline_smoke ^
  --model yolo11n.pt ^
  --imgsz 640 ^
  --batch 4 ^
  --baseline-epochs 5 ^
  --short-epochs 3 ^
  --final-epochs 5 ^
  --top-k 3 ^
  --workers 0 ^
  --device 0
```

Dry-run 不执行训练，只写计划、命令和 JSON 审计文件：

```bash
python scripts/run_diagnostic_augmentation_pipeline.py ^
  --dataset-root dataset ^
  --data-yaml dataset\data.yaml ^
  --output-dir outputs\diagnostic_aug_pipeline_dryrun ^
  --model yolov8n.pt ^
  --dry-run ^
  --skip-final-train ^
  --workers 0
```

关键输出：

- `baseline/baseline_metrics.json`
- `validation_prediction/validation_predictions.json`
- `diagnosis/diagnosis.json`
- `diagnosis/diagnosis_summary.md`
- `policies/candidate_policies.json`
- `proxy/proxy_metrics.json`
- `proxy/proxy_ranking.json`
- `short_training/short_train_results.json`
- `short_training/selected_policy.json`
- `dataset_builder/dataset_build_report.json`
- `final_training/final_train_metrics.json`
- `final_training/final_val_metrics.json`
- `report/experiment_summary.md`
- `report/experiment_summary.json`
- `report/tables_for_paper.md`

## 核心代码结构

```text
AutoAugment/
  augmentations/              # 可注册增强算子
  bbox/                       # bbox 转换、裁剪、IoU、仿射变换
  datasets/                   # 数据集读取
  diagnostics/                # 验证错误分析和增强建议
  diagnostic_pipeline/        # 诊断驱动增强闭环
  policies/                   # Policy / OperationSpec / search space
  search/                     # random search、proxy metrics、YOLO evaluator
  utils/                      # YOLO 数据集工具和输出工具
scripts/
  run_diagnostic_augmentation_pipeline.py
docs/
  diagnostic_augmentation_framework.md
  experiment_protocol.md
```

## 测试

```bash
pytest -q
```

所有 YOLO 命令在 Windows 下默认 `workers=0`。流水线不会默认启动 50 epoch 或更大规模训练。
