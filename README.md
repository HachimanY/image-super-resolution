# SRCNN 图像超分辨率

## 项目简介

本项目复现了 2014 年 Dong 等人的经典论文 **"Learning a Deep Convolutional Network for Image Super-Resolution"**，基于卷积神经网络（CNN）实现图像超分辨率重建，将低分辨率图像映射为高分辨率图像。

- [论文 PDF 下载](https://github.com/luzhixing12345/image-super-resolution/releases/download/v0.0.2/Learning.a.Deep.Convolutional.Network.for.Image.Super-Resolution.pdf)

| 低分辨率 | 高分辨率 |
|:--:|:--:|
| <img src="https://raw.githubusercontent.com/learner-lu/picbed/master/QQ%E6%88%AA%E5%9B%BE20220112003016.png"> | <img src="https://raw.githubusercontent.com/learner-lu/picbed/master/2.png"> |

## 参考资料

- [CSDN 博客：超分辨率详解](https://blog.csdn.net/qianbin3200896/article/details/104181552)
- [SRCNN-pytorch](https://github.com/yjn870/SRCNN-pytorch) — 本项目的参考实现
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) — 推荐的前沿超分辨率项目

---

## 环境安装

- Python 3.x
- PyTorch（根据 CUDA 版本单独安装：<https://pytorch.org/get-started/locally/>）

```bash
pip install -r requirements.txt
```

依赖项：`numpy`、`h5py`、`opencv-python`、`matplotlib`

---

## 项目结构

```
image-super-resolution/
  train.py          ← 训练入口
  use.py            ← 推理入口（支持 --compare 生成对比图）
  experiment.py     ← 参数对比实验脚本
  demo.py           ← GUI 演示系统
  model.py          ← SRCNN 网络定义（3层卷积）
  datasets.py       ← HDF5 数据集加载
  utils.py          ← 工具函数 + 可视化（训练曲线、对比图）
  requirements.txt  ← Python 依赖
  datasets/         ← HDF5 数据文件
  model/            ← 训练好的权重文件
```

---

## 第一步：准备数据集

下载标准 HDF5 数据集，放入 `./datasets/` 目录。

| 数据集 | 放大倍数 | 用途 | 下载链接 |
| ------ | -------- | ---- | -------- |
| 91-image | x2 | 训练集 | [下载](https://github.com/learner-lu/image-super-resolution/releases/download/v0.0.1/91-image_x2.h5) |
| 91-image | x3 | 训练集 | [下载](https://github.com/learner-lu/image-super-resolution/releases/download/v0.0.1/91-image_x3.h5) |
| 91-image | x4 | 训练集 | [下载](https://github.com/learner-lu/image-super-resolution/releases/download/v0.0.1/91-image_x4.h5) |
| Set5 | x2 | 测试集 | [下载](https://github.com/learner-lu/image-super-resolution/releases/download/v0.0.1/Set5_x2.h5) |
| Set5 | x3 | 测试集 | [下载](https://github.com/learner-lu/image-super-resolution/releases/download/v0.0.1/Set5_x3.h5) |
| Set5 | x4 | 测试集 | [下载](https://github.com/learner-lu/image-super-resolution/releases/download/v0.0.1/Set5_x4.h5) |

下载相同放大倍数的 91-image 和 Set5 各一个，放入 `./datasets/` 目录，例如 `./datasets/91-image_x2.h5` 和 `./datasets/Set5_x2.h5`。

---

## 第二步：训练模型

### 快速开始（默认参数）

```bash
python train.py
```

默认配置：x4 放大、学习率 1e-4、batch_size=32、训练 100 轮。

### 自定义参数

```bash
python train.py \
  --train-file 4 \
  --eval-file 4 \
  --batch-size 64 \
  --lr 1e-5 \
  --num-workers 8 \
  --epoch 500 \
  --f 10 \
  --save-history ./model/train_history.json
```

### 参数说明

| 参数 | 默认值 | 说明 |
| ---- | ------ | ---- |
| `--train-file` | 4 | 训练集放大倍数（2/3/4），对应 `datasets/91-image_x{N}.h5` |
| `--eval-file` | 4 | 测试集放大倍数（2/3/4），对应 `datasets/Set5_x{N}.h5` |
| `--batch-size` | 32 | 批大小 |
| `--num-workers` | 4 | 数据加载线程数（Windows 报错时设为 0） |
| `--lr` | 1e-4 | 学习率 |
| `--epoch` | 100 | 训练轮数 |
| `--f` | 5 | 每隔多少轮测试一次 |
| `--model-dir` | `./model` | 模型保存目录 |
| `--save-history` | `./model/train_history.json` | 训练历史保存路径 |

### 训练输出

- 最佳模型权重：`./model/best.pth`
- 训练历史记录：`./model/train_history.json`（包含每轮 loss 和 PSNR）

---

## 第三步：参数对比实验

自动运行多组不同参数配置，对比重建效果。

### 快速验证（每组 20 轮）

```bash
python experiment.py --quick
```

### 完整实验

```bash
python experiment.py --epochs 100
```

### 自定义输出目录

```bash
python experiment.py --epochs 100 --output-dir ./my_experiments
```

### 内置实验配置

| 编号 | 放大倍数 | 学习率 | 批大小 |
| ---- | -------- | ------ | ------ |
| 0 | x2 | 1e-4 | 32 |
| 1 | x4 | 1e-4 | 32 |
| 2 | x4 | 1e-3 | 32 |
| 3 | x4 | 1e-5 | 32 |
| 4 | x4 | 1e-4 | 64 |

### 实验输出

所有结果保存在 `./experiments/` 目录：

```
experiments/
  exp_0.json              ← 实验 0 的训练历史
  exp_0_curves.png        ← 实验 0 的训练曲线图
  exp_1.json
  exp_1_curves.png
  ...
  comparison.png           ← 多组实验叠加对比图
  summary.json             ← 所有实验排名汇总
```

终端输出排名表：

```
======================================================================
EXPERIMENT RESULTS SUMMARY
======================================================================
#    Scale  LR         Batch   Best PSNR  Best Epoch
----------------------------------------------------------------------
1    x4     0.0001     32         28.45          85
0    x2     0.0001     32         30.12          72
...
======================================================================
```

---

## 第四步：下载预训练模型（可选）

如果你已经自己训练了模型，可跳过此步。

- [x2 放大模型](https://github.com/learner-lu/image-super-resolution/releases/download/v0.0.1/b.pth)
- [x4 放大模型](https://github.com/learner-lu/image-super-resolution/releases/download/v0.0.1/best.pth)

将权重文件放入 `./model/` 目录，例如 `./model/best.pth` 或 `./model/b.pth`。

---

## 第五步：超分辨率推理

### 基本用法

```bash
python use.py --image path/to/image.jpg --scale 2
```

### 生成对比图（低分辨率 / 双三次插值 / SRCNN 三宫格）

```bash
python use.py --image path/to/image.jpg --scale 2 --compare
```

### 推理参数

| 参数 | 默认值 | 说明 |
| ---- | ------ | ---- |
| `--image` | （必填） | 输入图片路径 |
| `--scale` | 2 | 放大倍数（2/3/4） |
| `--weights-file` | `./model/best.pth` | 模型权重路径 |
| `--compare` | 关闭 | 生成三宫格对比图 |

### 输出文件

- 超分辨率结果：`image_srcnn_x2.jpg`
- 对比图：`image_comparison.png`（使用 `--compare` 时生成）

---

## 第六步：GUI 演示系统

```bash
python demo.py
```

打开图形界面后操作步骤：

1. **Model Weights** — 点击 Browse 选择模型权重文件（默认 `./model/best.pth`）
2. **Input Image** — 点击 Browse 选择要处理的图片
3. **Scale Factor** — 选择放大倍数（2 / 3 / 4）
4. 点击 **Run Super-Resolution** 按钮
5. 等待处理完成，界面显示 PSNR 值和输出文件路径

---

## 常见问题

### OSError: Unable to open file（HDF5 文件锁错误）

```bash
# Linux
export HDF5_USE_FILE_LOCKING='FALSE'

# 或永久写入 ~/.bashrc
echo 'export HDF5_USE_FILE_LOCKING="FALSE"' >> ~/.bashrc
source ~/.bashrc
```

### OSError: [WinError 1455] 页面文件太小（Windows）

```bash
python train.py --num-workers 0
```

### FutureWarning: torch.load with weights_only=False

这是 PyTorch 的弃用警告，不影响功能，可忽略。

---

## 总结

本项目实现了完整的图像超分辨率处理流程，包含：

- **模型训练** — 可配置参数，自动记录训练历史
- **参数对比实验** — 自动运行多组配置，生成排名汇总
- **可视化分析** — 训练曲线图、定量指标（PSNR）、视觉对比图
- **GUI 演示系统** — 图形界面一键超分辨率

SRCNN 架构简洁（3 层卷积），适合作为学习入门的基线模型。如需更高质量，可参考 EDSR、RCAN 或 [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) 等更深的网络架构。
