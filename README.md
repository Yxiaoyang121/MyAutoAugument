# MyAutoAugument

MyAutoAugument 是一个面向工业图像数据集的自动增强框架，支持模块化增强算子、策略组合、轻量模型评估和基于反馈的进化搜索。

## 架构摘要

```text
src/
  augmentations/   核心增强算子与注册表
  policies/        增强策略定义、随机采样、交叉与变异
  models/          轻量评估模型与策略评分器
  search/          进化搜索与反馈优化闭环
  utils/           图像校验、随机种子、数据集、路径和演示数据工具
```

## 核心能力

- 统一增强接口：`apply(image, params)`
- 11 个标准增强算子：亮度、对比度、模糊、高斯噪声、旋转、翻转、平移、透视变换、伽马校正、饱和度、反光/光照变化
- 策略表达：每个策略由多个增强步骤组成，每步包含算子名、执行概率、强度和附加参数
- 策略搜索：内置简单进化搜索，支持种群初始化、精英保留、交叉、变异和固定种子
- 反馈闭环：策略 → 增强训练集 → 轻量模型训练 → 验证准确率 → 更新策略
- 兼容旧接口：`src/AugumentMethods.py` 保留旧函数名，便于已有脚本迁移

## 快速开始

安装依赖：

```bash
pip install -r requirements.txt
```

运行增强示例：

```bash
python examples/AugumentMethodsTest.py
```

运行策略搜索闭环：

```bash
python examples/run_policy_search.py --generations 3 --population-size 6 --seed 42
```

运行测试：

```bash
pytest
```

## 策略示例

```python
from src.policies.policy import AugmentationStep, Policy

policy = Policy(
    [
        AugmentationStep("brightness", probability=0.5, strength=0.8),
        AugmentationStep("blur", probability=0.3, strength=0.2),
    ]
)
```

## 优化闭环

1. `policies` 随机生成或接收初始增强策略。
2. `augmentations` 根据策略概率和强度处理训练图像。
3. `models` 使用增强后的训练集训练轻量最近质心分类器。
4. 验证集准确率作为当前策略奖励分数。
5. `search` 保留高分策略，并通过交叉、变异生成下一代策略。
6. 多轮迭代后输出最佳策略和搜索历史。

## 数据集格式

真实数据可按类别分目录组织，并通过 `src.utils.dataset.load_image_folder` 读取。传入路径应使用相对项目根目录的路径。

```text
dataset/
  class_a/
    image_001.png
  class_b/
    image_002.png
```
