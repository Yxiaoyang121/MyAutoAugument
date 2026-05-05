# MyAutoAugument

## 项目简介
MyAutoAugument 提供了一套多功能的图像数据增强算法，解决工业视觉检测中的多种不确定性问题，包括机械位姿偏差、光源波动、反射率差异、粉尘干扰等。

## 快速开始

1. 克隆仓库：

    git clone https://github.com/Yxiaoyang121/MyAutoAugument.git

2. 安装依赖：

    pip install -r requirements.txt

3. 运行示例：

    python examples/AugumentMethodsTest.py


## 项目结构

```
MyAutoAugument/
├── src/                  # 核心代码文件
│   └── AugumentMethods.py
├── tests/                # 单元测试（pytest）
│   └── test_AugumentMethods.py
├── examples/             # 示例运行脚本
│   └── AugumentMethodsTest.py
├── assets/               # 图片资源
│   └── Luntai1.jpg (请确保存在)
├── README.md
├── requirements.txt
```

## 贡献
欢迎提交 PR。