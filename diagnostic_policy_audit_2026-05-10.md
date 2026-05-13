# Diagnostic Policy Audit - 2026-05-10

审计日期：2026-05-11  
审计范围：2026-05-10 生成或修改的训练记录，以及当前代码中 validation analyzer / policy updater / policy search 逻辑。  
约束：本次只做日志读取和静态分析，未启动新的 YOLO 训练，未删除已有结果，未修改项目代码。

## 1. 找到的昨天训练记录路径

### 1.1 主要 policy search 训练记录

1. `outputs/my_policy_search_train_yolo_tiled_50`
   - 时间：2026-05-09 23:49 启动，2026-05-10 03:18/15:06 更新结果。
   - 配置：`outputs/my_policy_search_train_yolo_tiled_50/run_config.json`
   - 汇总：`outputs/my_policy_search_train_yolo_tiled_50/trials.csv`
   - 汇总 JSON：`outputs/my_policy_search_train_yolo_tiled_50/trials.json`
   - 最优策略：`outputs/my_policy_search_train_yolo_tiled_50/best_policy.json`
   - trial 数：50
   - 每个 trial：`outputs/my_policy_search_train_yolo_tiled_50/trials/trial_000` 到 `trial_049`
   - 每轮日志：`policy.json`、`metrics.json`、`train_stdout.log`、`train_stderr.log`、`val_stdout.log`、`val_stderr.log`、`train_runs/train/results.csv`、`weights/best.pt`
   - YOLO 输出目录：`runs/detect/val-53` 到 `runs/detect/val-100`
   - 配置摘要：`evaluator=train_yolo`，`metric=map50`，`trials=50`，`samples=100`，`epochs=10`，`imgsz=640`，`batch=4`，`hybrid_proxy_weight=0.2`
   - 最优 trial：`trial_007`，score `0.9684666667`

2. `outputs/advisor_policy_search_map5095`
   - 时间：2026-05-10 19:00 启动，2026-05-10 22:27 完成。
   - 配置：`outputs/advisor_policy_search_map5095/run_config.json`
   - 汇总：`outputs/advisor_policy_search_map5095/trials.csv`
   - 汇总 JSON：`outputs/advisor_policy_search_map5095/trials.json`
   - 最优策略：`outputs/advisor_policy_search_map5095/best_policy.json`
   - trial 数：50
   - 每个 trial：`outputs/advisor_policy_search_map5095/trials/trial_000` 到 `trial_049`
   - 每轮日志：`policy.json`、`metrics.json`、`train_stdout.log`、`train_stderr.log`、`val_stdout.log`、`val_stderr.log`、`train_runs/train/results.csv`、`weights/best.pt`
   - YOLO 输出目录：`runs/detect/val-101` 到 `runs/detect/val-150`
   - 配置摘要：`evaluator=train_yolo`，`metric=map50_95`，`trials=50`，`samples=90`，`epochs=10`，`imgsz=640`，`batch=4`，`hybrid_proxy_weight=0.0`
   - 使用诊断搜索空间：`outputs/diagnostics/baseline_error_analysis/advisor_search_space.json`
   - 最优 trial：`trial_002`，score `0.616`

### 1.2 独立诊断记录

1. `outputs/diagnostics/baseline_error_analysis`
   - 时间：2026-05-10 18:56。
   - 关键文件：
     - `error_summary.json`
     - `error_summary.csv`
     - `per_image_errors.csv`
     - `per_object_errors.csv`
     - `diagnosis_report.txt`
     - `augmentation_advice.json`
     - `augmentation_advice.txt`
     - `advisor_search_space.json`
     - `predict_stdout.log`
     - `predict_stderr.log`
     - `predict_command.txt`
   - 诊断摘要：GT=19，TP=19，FP=2，FN=0，localization_weak=0，precision=0.9048，recall=1.0。
   - 实际 advice：`issues=[]`，结论为无 dominant failure mode，所有 operation 权重均为 1.0，仅保留 forbidden combinations。

### 1.3 baseline / best policy 对比记录

1. `outputs/baseline_compare_tiled`
   - 时间：2026-05-10 16:20 到 16:44。
   - 文件：
     - `run_config.json`
     - `baseline_no_aug/metrics.json`
     - `best_policy_aug/applied_policy.json`
     - `best_policy_aug/metrics.json`
     - `compare_metrics.csv`
     - `compare_metrics.json`
   - 对比结果：
     - `baseline_no_aug`: mAP50=0.995, mAP50-95=0.659, precision=0.941, recall=1.0
     - `best_policy_aug`: mAP50=0.995, mAP50-95=0.658, precision=0.979, recall=0.99

2. `outputs/advisor_baseline_compare_tiled`
   - 时间：2026-05-10 22:46 到 22:57。
   - 文件：
     - `run_config.json`
     - `baseline_no_aug/metrics.json`
     - `advisor_best_policy_aug/applied_policy.json`
     - `advisor_best_policy_aug/metrics.json`
     - `compare_metrics.csv`
     - `compare_metrics.json`
   - 对比结果：
     - `baseline_no_aug`: mAP50=0.995, mAP50-95=0.659, precision=0.941, recall=1.0
     - `advisor_best_policy_aug`: mAP50=0.995, mAP50-95=0.618, precision=0.889, recall=0.959

## 2. Validation analyzer 结论：部分通过

结论：部分通过。

当前 analyzer 不只是输出 mAP、precision、recall、loss。`AutoAugment/diagnostics/yolo_error_analysis.py` 能从 YOLO prediction labels 和 GT labels 做对象级匹配，并输出：

- `TP`
- `FN`
- `FP`
- `localization_weak`
- `by_class`
- `by_size`
- `by_position`
- `quality.false_negatives`
- `quality.false_positives`
- `fp_background`
- ROI 层面的 `brightness_mean`、`contrast_std`、`low_contrast`、`too_dark`、`too_bright`

对应代码位置：

- `analyze_yolo_errors`: `AutoAugment/diagnostics/yolo_error_analysis.py:21`
- `match_detections`: `AutoAugment/diagnostics/yolo_error_analysis.py:117`
- `error_type=TP/FN/FP/localization_weak`: `AutoAugment/diagnostics/yolo_error_analysis.py:157`, `188`, `207`, `227`
- `summarize_error_records`: `AutoAugment/diagnostics/yolo_error_analysis.py:335`
- `diagnosis_report.txt` 输出：`AutoAugment/diagnostics/yolo_error_analysis.py:492`, `541`

不足：

- 不分析训练/验证 loss 曲线，因此不能判断过拟合。
- 不做跨 epoch 趋势诊断。
- 不直接检测模糊或噪声敏感性，只通过低对比、过暗、过亮和错误分布间接推断。
- `diagnosis_report.txt` 文本只写总体、类别和尺寸诊断，位置、质量、FP 背景信息主要保存在 JSON/CSV 中，文本报告不完整。
- 2026-05-10 的实际诊断没有发现 dominant issue：`augmentation_advice.json` 中 `issues=[]`。

## 3. Policy updater 结论：半闭环，实际 2026-05-10 advisor run 退化为近似随机

总体结论：半闭环。

代码中存在“诊断原因 -> 搜索空间调整”的映射，但不是每轮 trial 后根据 validation diagnosis 更新下一轮 policy。流程是：

1. 先运行一次 baseline/error analysis。
2. `generate_augmentation_advice(error_summary)` 根据错误摘要生成 `advisor_search_space.json`。
3. `run_policy_search.py --search-space-json ...` 加载该搜索空间。
4. `RandomSearch` 仍然随机采样 policy，只是采样分布可能被诊断权重影响。

存在的映射逻辑：

- 小目标漏检高：`small_object_fn_high`，提高 `sharpen` / `clahe` / `scale` / `translate`，降低 blur/noise 强度。
- 边缘目标漏检高：`edge_object_fn_high`，提高 `translate` / `affine` / `scale`。
- 低对比漏检高：`low_contrast_fn_high`，提高 `contrast` / `gamma` / `clahe` / `sharpen`。
- 曝光问题：`exposure_fn_high`，提高 `brightness` / `gamma`。
- 类别召回不均衡：`class_recall_imbalance`，给出 class-aware sampling 建议。
- 误检偏高：`false_positive_high`，降低 `gaussian_noise` / `cutout`，建议 hard negative tiles。

对应代码位置：

- advice 入口：`AutoAugment/diagnostics/advisor.py:24`
- 各 issue 映射：`AutoAugment/diagnostics/advisor.py:46`, `67`, `85`, `97`, `116`, `127`
- 输出 `advisor_search_space`: `AutoAugment/diagnostics/advisor.py:137`
- 加载 advisor JSON：`AutoAugment/policies/search_space.py:94`
- 权重采样：`AutoAugment/policies/search_space.py:80`
- 每轮随机采样：`AutoAugment/search/random_search.py:89`

为什么不是“真闭环”：

- `RandomSearch` 每轮只执行 `sample_policy -> evaluate -> write results`。
- 代码没有读取本轮 `diagnosis`、`error_type`、`failure_reason` 或 `val_analysis` 来更新下一轮策略。
- 代码没有 `before_policy -> diagnosis -> adjust_reason -> after_policy` 的 updater。
- `advisor.py` 是一次性启发式 search-space advisor，不是在线 policy updater。

2026-05-10 的实际情况更弱：

- `outputs/diagnostics/baseline_error_analysis/augmentation_advice.json` 中 `issues=[]`。
- `advisor_search_space.operation_weights` 全部为 `1.0`。
- 因此 `outputs/advisor_policy_search_map5095` 虽然使用了 `--search-space-json`，但实际主要仍是随机搜索，只多了 forbidden combinations 约束。

## 4. 每轮记录完整性：不完整

现有每轮记录完整保存了：

- `trial_index`
- `score`
- `policy`
- `trial_dir`
- `metrics`
- `policy.json`
- `metrics.json`
- YOLO train / val command
- train / val stdout/stderr log
- `best.pt`
- `data.yaml`

两个主要 search run 均有 50 个 trial dir 和 50 个 `metrics.json`。

但按项目目标所需字段审计：

| 字段 | 当前是否记录 |
| --- | --- |
| `trial_id` / `trial_index` | 有，字段名为 `trial_index` |
| `before_policy` | 没有 |
| `val_metrics` | 部分有，保存在 `metrics.yolo_map50` / `metrics.yolo_map50_95` / `final_score` |
| `diagnosis` | 没有逐 trial 诊断 |
| `adjust_reason` | 没有 |
| `after_policy` | 没有 |
| `policy_diff` | 没有 |
| `accepted/rejected` | 没有 |

直接证据：

- `trials.csv` 字段只有 `trial_index, score, trial_dir, policy_name, metrics_json, policy_json`。
- `trials.json` 每项字段只有 `trial_index, score, metrics, policy, trial_dir`。
- `rg` 在两个主要 search 输出中没有找到 `before_policy`、`after_policy`、`policy_diff`、`adjust_reason`、`accepted`、`rejected`、`val_analysis`、`failure_reason`。

## 5. 是否支持 short-train：部分支持

结论：部分支持。

支持点：

- `examples/run_policy_search.py` 支持 `--epochs`、`--samples`、`--imgsz`、`--batch`。
- `YoloTrainValEvaluator` 每个 trial 可以用较小 epoch 训练。
- `ProxyEvaluator` 支持不训练模型的代理评价。
- 搜索阶段和最终对比阶段事实上分离：`run_policy_search.py` 做搜索，`tools/compare_yolo_baseline.py` 用保存的 policy 做 baseline 对比。
- 2026-05-10 的搜索阶段使用 `epochs=10`；baseline compare 使用 `epochs=30`。

不足：

- 没有显式 `search_epochs`、`short_epochs`、`proxy_epochs`、`final_epochs` 配置命名。
- 没有内建“搜索阶段 short-train，最终阶段 full-train”的统一 pipeline 状态记录。
- 没有 `freeze` 参数。
- 没有独立的 `subset` 参数，当前用 `--samples` 控制每个 trial 采样数量。
- 没有记录“这是 proxy / short train / final train”的阶段字段。

对应代码位置：

- CLI 参数：`examples/run_policy_search.py:34`, `35`, `49`, `50`, `51`
- short train 提示：`examples/run_policy_search.py:234`
- evaluator 参数：`AutoAugment/search/evaluator.py:298`, `313`, `314`, `315`
- baseline compare 默认 full-ish 配置：`tools/compare_yolo_baseline.py:40`

## 6. 是否能限制 trial 到 8~12：可以，但昨天实际用了 50

结论：可以限制。

支持点：

- CLI 参数 `--trials` 默认 10：`examples/run_policy_search.py:34`
- `RandomSearch(num_trials=...)` 控制循环次数：`AutoAugment/search/random_search.py:49`, `66`, `89`
- 因此可以直接设置 `--trials 8`、`--trials 10` 或 `--trials 12`。

不足：

- `examples/run_diagnostic_aug_pipeline.py` 默认 `--trials 50`，推荐命令也打印 50。
- 没有 `search_budget`、`patience`、`early_stop`、`top_k`。
- 代码中出现的 `max_trials` 是 crop transform 内部尝试次数，不是 policy search trial budget。

2026-05-10 实际记录：

- `outputs/my_policy_search_train_yolo_tiled_50`: `trials=50`
- `outputs/advisor_policy_search_map5095`: `trials=50`

## 7. 最小修改清单

### P0

1. 增加逐 trial 诊断记录 schema。
   - 每轮至少写入 `trial_id`、`before_policy`、`val_metrics`、`diagnosis`、`adjust_reason`、`after_policy`、`policy_diff`、`accepted/rejected`。
   - 如果仍使用 random search，也应明确写 `adjust_reason: random_sample_from_advisor_space`，避免伪装成闭环。

2. 增加真正的 policy updater。
   - 输入：上一轮 policy、val metrics、错误诊断。
   - 输出：下一轮 policy 或下一轮 search-space 权重。
   - 需要显式读取 `diagnosis` / `error_type` / `val_analysis`，而不是只读取一次 baseline `advisor_search_space.json`。

3. 把 advisor run 的默认 trial 数改为 8~12。
   - `examples/run_diagnostic_aug_pipeline.py` 当前默认 50，和轻量级搜索目标不一致。

### P1

1. 把 `diagnosis_report.txt` 补全为完整文本报告。
   - 增加 `by_position`、`quality.false_negatives`、`quality.false_positives`、`fp_background`。

2. 增加阶段配置命名。
   - 推荐新增 `search_epochs` / `final_epochs` 或在 run_config 中明确 `stage=search_short_train/final_train`。

3. 增加 `freeze` 和 `subset` 支持。
   - `--samples` 只能控制采样数量，不等价于可复现实验 subset 定义。

4. 增加 early stop / patience。
   - 轻量搜索可在连续若干 trial 无提升时停止。

### P2

1. 增加 overfitting 诊断。
   - 从 `results.csv` 读取 train/val loss 曲线，输出 train-val gap 和趋势。

2. 增加模糊/噪声敏感性诊断。
   - 当前只有低对比、亮度相关指标，不能直接说明 blur/noise sensitivity。

3. 增加 policy diff 可视化。
   - 例如 `policy_history.jsonl` 和 `policy_diff.csv`，方便论文中展示每轮策略调整轨迹。

4. 增加 baseline-advisor 对比自动摘要。
   - 当前已有 compare 输出，但没有把“搜索最优 policy 是否显著优于 no-aug baseline”反馈给 updater。

## 8. 总体判断

当前项目已经具备“诊断驱动搜索空间”的雏形，不是普通纯随机 AutoAugment 的最原始形态；但它还不是“基于验证误差诊断的自适应闭环调整”。更准确的表述是：

> 一次性验证误差诊断驱动的加权随机增强策略搜索。

如果论文目标要强调“每轮基于验证误差诊断自适应调整增强策略”，当前实现证据不足。最小可发表闭环应至少补齐逐轮 diagnosis、adjust_reason、policy_diff 和 accepted/rejected，并让 updater 在 trial 之间真实消费诊断结果。
