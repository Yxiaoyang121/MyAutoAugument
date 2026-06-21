# Backend Capability Map

审计日期：2026-05-13  
审计范围：`AutoAugment/`、`demo/`、`docs/`、`examples/`、`tests/`、`tools/`、`README.md`、`diagnostic_policy_audit_2026-05-10.md`。  
原则：本表只列当前代码中真实存在的类、函数、registry 或 CLI 能力，不列未实现增强方法。

## 1. 已实现的数据增强方法清单

后端真实 registry 位于 `AutoAugment/augmentations/registry.py`，注册项来自 `AutoAugment/augmentations/ops.py`。所有方法接收 `image, labels, bboxes, params, strength, rng`，并返回增强后的 `image, labels, bboxes`。

说明：`prob` 和 `strength` 是 `Policy.OperationSpec` 的通用字段，默认值均为 `1.0`，后端会裁剪到 `[0, 1]`。下表的参数范围是 GUI 暴露的后端兼容范围；当前后端对大多数 `params` 不做强校验，而是在使用时转换为 `float/int`。

| 方法 | 类别 | 参数名 / 默认值 / 范围 | bbox 支持 | 是否改变 bbox | 适合检测 |
| --- | --- | --- | --- | --- | --- |
| `brightness` | 光照/颜色 | `max_delta=0.25`，建议 `0..1` | 是 | 否 | 是 |
| `contrast` | 光照/颜色 | `max_delta=0.5`，建议 `0..2` | 是 | 否 | 是 |
| `gamma` | 光照/颜色 | `min_gamma=0.7`、`max_gamma=1.5`，建议 `0.05..5` | 是 | 否 | 是 |
| `gaussian_noise` | 噪声 | `std` 或 `max_std=0.08`，建议 `0..1` | 是 | 否 | 是 |
| `salt_pepper_noise` | 噪声 | `amount` 或 `max_amount=0.02`，建议 `0..1` | 是 | 否 | 是 |
| `gaussian_blur` | 工业偏差 | `max_kernel=9`，建议奇数 `1..31`；`sigma` 可选 | 是 | 否 | 是 |
| `motion_blur` | 工业偏差 | `max_kernel=11`，建议奇数 `1..31`；`angle` 可选 `0..180` | 是 | 否 | 是 |
| `median_blur` | 工业偏差 | `max_kernel=7`，建议奇数 `1..31` | 是 | 否 | 是 |
| `sharpen` | 工业偏差 | `amount=1.2`，建议 `0..5`；`sigma=1.0`，建议 `0..10` | 是 | 否 | 是 |
| `clahe` | 工业偏差 | `max_clip_limit=4.0`，建议 `1..20`；`tile_grid_size=(8,8)` | 是 | 否 | 是 |
| `cutout` | 工业偏差 | `max_holes=3`，建议 `1..32`；`max_fraction=0.25`，建议 `0..1`；`fill_value` 可选 | 是 | 否 | 是 |
| `random_erasing` | 工业偏差 | 同 `cutout`，后端直接调用 `cutout` | 是 | 否 | 是 |
| `horizontal_flip` | 几何 | 无额外 params | 是 | 是 | 是 |
| `vertical_flip` | 几何 | 无额外 params | 是 | 是 | 是 |
| `rotate` | 几何 | `max_angle=15.0`，建议 `0..180`；`angle` 可选固定角度 | 是 | 是 | 是 |
| `translate` | 几何 | `max_translate=0.1`，建议 `0..1`；`dx/dy` 可选 | 是 | 是 | 是 |
| `scale` | 几何 | `max_delta=0.25`，建议 `0..1`；`scale` 可选固定缩放 | 是 | 是 | 是 |
| `affine` | 几何 | `max_angle=10.0`、`max_scale_delta=0.15`、`max_shear=5.0`、`max_translate=0.08`；`angle/scale/shear_x/shear_y/tx/ty` 可选 | 是 | 是 | 是 |
| `crop` | 几何 | `max_crop_fraction=0.4`，建议 `0..0.95`；`min_visibility=0.2`，建议 `0..1`；`crop_box/crop_size/width_range/height_range` 可选 | 是 | 是 | 是 |
| `resize_letterbox` | 几何 | `target_size` 或 `width/height` 可选；`color=(114,114,114)` | 是 | 是 | 是 |

## 2. 数据集格式支持情况

| 格式 | 当前支持程度 | 代码证据 |
| --- | --- | --- |
| YOLO | 完整工作流支持：label 读写、dataset record 查找、train/val split、policy search、apply policy、诊断分析 | `AutoAugment/formats/yolo.py`、`AutoAugment/utils/yolo_dataset.py`、`examples/run_policy_search.py` |
| COCO | annotation 与内部 sample 的转换；有 COCO dataset 模块 | `AutoAugment/formats/coco.py`、`AutoAugment/datasets/coco_dataset.py` |
| VOC | XML 读取与 bbox 转换辅助 | `AutoAugment/formats/voc.py`、`AutoAugment/bbox/convert.py` |

Qt Dataset 页面当前按需求实现 YOLO 目录检查：`images/train`、`images/val`、`labels/train`、`labels/val`、`data.yaml`。

## 3. bbox 变换能力

已实现能力：

- 内部 bbox 统一为 `xyxy`。
- YOLO normalized `xywh` 与 `xyxy` 互转。
- COCO `xywh` 与 `xyxy` 互转。
- VOC `xyxy` passthrough。
- 仿射矩阵作用于 bbox 四角，再取外接矩形。
- bbox 裁剪到图像边界。
- 按最小宽、高、面积过滤无效 bbox。
- crop 后按 `min_visibility` 过滤可见面积不足 bbox。
- flip、rotate、translate、scale、affine、crop、resize_letterbox 均同步更新 bbox。

## 4. 评估指标

`ProxyEvaluator` 和 YOLO evaluator 已实现或解析的指标包括：

- 图像与 bbox：`image_valid_rate`、`bbox_valid_rate`、`bbox_safe_rate`、`bbox_retention`、`bbox_retention_raw`
- 小目标/边缘/类别：`tiny_target_retention`、`small_target_retention`、`edge_target_retention`、`class_coverage_after`、`rare_class_retention`
- 曝光与多样性：`exposure_score`、`variance_score`、`exposure_diversity_score`、`strength_penalty`、`diversity_score`
- 综合代理分：`proxy_score`、`proxy_score_components`
- YOLO：`yolo_map50`、`yolo_map50_95`、`yolo_metric_score`、`final_score`
- 诊断对象级统计：`TP`、`FP`、`FN`、`localization_weak`、`precision`、`recall`、`fn_rate`

## 5. 策略搜索 / 闭环优化能力

已实现：

- `Policy` / `OperationSpec` JSON round-trip。
- `SearchSpace` / `OperationSpace` 随机采样。
- 默认 detection search space。
- 从 advisor JSON 加载 `allowed_operations`、`operation_weights`、`strength_ranges`、`prob_ranges`、`forbidden_combinations`。
- `RandomSearch` 多 trial policy search。
- `ProxyEvaluator` 快速代理评估。
- `YoloCommandEvaluator` 使用已有 YOLO val 命令解析 mAP。
- `YoloTrainValEvaluator` 每个 trial 训练 YOLO，并在固定 val 上验证 mAP。
- `ProxyPrefilterConfig` 支持候选 policy 预筛选。
- 当前 `RandomSearch` 已记录 `before_policy`、`after_policy`、`policy_diff`、`adjust_reason`、`accepted/rejected`、per-trial `diagnosis.json`。
- `DiagnosticPolicyUpdater` 会根据诊断结果更新下一轮 search space 权重与 prob/strength range。

限制：

- 不是强化学习；诊断与更新是启发式规则。
- 真 YOLO 训练依赖本机 `yolo` 命令、模型权重和环境。
- backend CLI `examples/run_policy_search.py` 不接收固定 `policy.json` 作为唯一 trial policy。Qt 新增的 GUI launcher 在 `gui/adapters/` 中把 Policy Editor 当前策略转换为 backend `SearchSpace`，不改核心算法。

## 6. 诊断能力

已实现诊断来源：

- `AutoAugment/diagnostics/yolo_error_analysis.py`
- `AutoAugment/diagnostics/advisor.py`
- `AutoAugment/search/adaptive.py`

可诊断或推断的问题：

- 小目标漏检：`small_object_fn_high`
- 边缘/位置问题：`edge_object_fn_high`、`position_fn_gap`
- 弱定位：`localization_weak`、`weak_localization_gap`
- 类别召回/精度不均衡：`class_recall_imbalance`、`class_precision_imbalance`
- 误检偏高：`false_positive_high`、`false_positive_precision_gap`
- 漏检/召回不足：`false_negative_recall_gap`
- 过暗/过亮：ROI `too_dark`、`too_bright`，advisor `exposure_*`
- 低对比：ROI `low_contrast`，advisor `low_contrast_*`
- bbox 安全/有效/保留不足：通过 proxy metrics 暴露
- 样本或 dominant issue 不明显：`stable_validation_keep_light_policy`

## 7. 输出文件

已有或由后端 search 写出的输出：

- 顶层：`run_config.json`、`stage_config.json`、`input_dataset.txt`、`README.txt`
- 汇总：`trials.json`、`trials.csv`、`policy_history.jsonl`、`policy_history.csv`、`policy_history.md`
- 最优：`best_policy.json`
- 每个 trial：`policy.json`、`metrics.json`、`diagnosis.json`、`trial_record.json`
- YOLO 训练/验证日志：`train_stdout.log`、`train_stderr.log`、`val_stdout.log`、`val_stderr.log`
- 诊断工具：`error_summary.json`、`error_summary.csv`、`per_image_errors.csv`、`per_object_errors.csv`、`diagnosis_report.txt`、`augmentation_advice.json`、`augmentation_advice.txt`、`advisor_search_space.json`

Qt GUI launcher 额外写出便于 GUI 读取的顶层文件：`summary.json`、`trial_record.json`、`policy.json`、`metrics.json`、`diagnosis.json`、`gui_experiment_config.json`、`gui_policy.json`、`gui_search_space.json`。

## 8. 已经接入 GUI 的能力

- Augmentation Library：从后端 registry 自动列出真实增强方法；展示分类、说明、参数、默认值、范围、bbox 影响。
- Policy Editor：从真实方法库添加操作；编辑 `prob`、`strength`、后端 params JSON；保存/加载 backend `policy.json`。
- Preview：读取真实图片和 YOLO label；调用后端 `apply_policy`；显示增强前后 bbox 与有效性。
- Dataset：检查 YOLO `images/train`、`images/val`、`labels/train`、`labels/val`、`data.yaml`；统计图片、label、类别、空 label、缺失 label、坏图。
- Experiment：使用 `ExperimentConfig` 合并 Dataset、Policy、Experiment 参数；写完整 config；通过 `LocalExperimentRunner` 启动 `gui/adapters/gui_experiment_launcher.py`，再调用后端 `RandomSearch`。
- Results：读取真实输出目录；支持 `summary.json`、`trial_record.json`、`diagnosis.json`、`policy.json`、`metrics.json`，也兼容后端 `trials.json`、`trials.csv`、`best_policy.json`。
- Diagnostics：优先读取 backend `diagnosis.json` / advisor 输出，按小目标、弱定位、类别、曝光、低对比、bbox、样本不足等分类展示。
- Logs / Outputs：展示输出目录结构，查看日志、JSON、CSV、policy、metrics、diagnosis，并显示最佳 trial / policy。

## 9. 尚未完全接入 GUI 的能力与原因

- COCO / VOC 数据集完整 GUI 工作流：后端目前主要是格式转换或读取辅助，policy search 和诊断闭环围绕 YOLO 实现。
- `CommandEvaluator` 自定义命令模板：Qt 当前主路径暴露 proxy 和 train_yolo，避免把任意 shell 模板作为桌面默认入口。
- 独立 `tools/tile_yolo_dataset.py` 长图切片：这是预处理工具，不是 augmentation registry 方法；未放入 Policy Editor，避免伪装成后端增强 op。
- baseline compare 工具：`tools/compare_yolo_baseline.py` 是独立实验后处理流程，当前 Logs/Outputs 可查看其结果，但没有做专门 wizard。
- final full-dataset training：现有后端显式标记未实现，GUI 不暴露为可运行闭环阶段。
