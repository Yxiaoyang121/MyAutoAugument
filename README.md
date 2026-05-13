# AutoAugment for Industrial Object Detection

这是一个面向工业视觉目标检测的数据增强框架，统一支持 YOLO / COCO / VOC 标注格式，并以 `sample -> sample` 作为唯一增强接口。

## 核心数据结构

```python
sample = {
    "image": image,
    "bboxes": bboxes,
    "labels": labels,
}
```

- `image`: `numpy.ndarray(H, W, C)`
- `bboxes`: `numpy.ndarray(N, 4)`，统一使用 `xyxy` 格式
- `labels`: `numpy.ndarray(N)`

## 项目结构

```text
AutoAugment/
  transforms/
    geometric/      水平翻转、垂直翻转、旋转、平移、缩放、随机裁剪
    color/          颜色增强预留模块
    blur/           模糊增强预留模块
    noise/          噪声增强预留模块
  bbox/
    affine.py       bbox 四角点仿射变换
    clip.py         bbox 越界裁剪与合法性过滤
    convert.py      YOLO / COCO / VOC 与 xyxy 转换
    iou.py          bbox IoU
  formats/
    yolo.py         YOLO txt 读写
    coco.py         COCO annotation 与 sample 转换
    voc.py          VOC XML 读取
  pipelines/
    compose.py      检测增强流水线
  visualize/
    draw_bbox.py    bbox 可视化
  datasets/
    yolo_dataset.py YOLO 检测数据集读取
    coco_dataset.py COCO 检测数据集读取
```

## 已实现增强

- `HorizontalFlip`
- `VerticalFlip`
- `Rotate`
- `Translate`
- `Scale`
- `RandomCrop`

这些增强都会同步变换 bbox，并在增强后执行越界裁剪和合法性检查，过滤宽高或面积无效的 bbox。

## 使用示例

```python
import numpy as np

from AutoAugment.pipelines import Compose
from AutoAugment.transforms.geometric import HorizontalFlip, Rotate, Translate

sample = {
    "image": np.zeros((640, 640, 3), dtype=np.uint8),
    "bboxes": np.asarray([[100, 120, 240, 260]], dtype=np.float32),
    "labels": np.asarray([1], dtype=np.int64),
}

pipeline = Compose(
    [
        HorizontalFlip(probability=0.5),
        Rotate(angle_range=(-5, 5), probability=0.5),
        Translate(dx_range=(-0.05, 0.05), dy_range=(-0.05, 0.05), probability=0.5),
    ],
    seed=42,
)

augmented = pipeline(sample)
```

## 单图调试

`demo/` 目录用于单张图像增强调试，支持读取 YOLO labels、保存增强图、保存增强后的 YOLO labels，并输出 bbox 可视化图。

不传参数时会使用内置检测样本直接运行：

```bash
python demo/single_image_augment.py
```

读取真实 YOLO 图像和标签：

```bash
python demo/single_image_augment.py --image path/to/image.jpg --label path/to/image.txt --show
```

## 批量增强 YOLO 数据集

`tools/` 目录用于批量增强 YOLO 数据集，内部使用 `Compose` pipeline。输出目录结构为：

```text
outputs/yolo_augmented/
  images/
  labels/
  visualize/
```

不传参数时会创建一个最小 YOLO 示例数据集并直接增强：

```bash
python tools/augment_yolo_dataset.py --visualize
```

增强真实 YOLO 数据集：

```bash
python tools/augment_yolo_dataset.py --images-dir dataset/images --labels-dir dataset/labels --output-dir outputs/my_augmented --repeat 2 --visualize
```

## 设计原则

- 不包含分类训练流程
- 不使用 ImageFolder
- 不使用 `(image, class_id)` 单标签结构
- 所有增强输入输出均为目标检测 `sample`
- 所有 bbox 均为 `xyxy`
- 数据集路径必须是相对项目根目录的路径
- 随机增强支持固定 seed，便于复现

## 验证

```bash
pytest -q
```
