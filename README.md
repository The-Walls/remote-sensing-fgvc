# Remote-Sensing Fine-Grained Ship Recognition

基于 PyTorch 的遥感舰船细粒度图像分类项目。主数据集为 FGSCR-42，任务是从单张遥感图像中识别 42 个舰船细粒度类别；当前项目不包含目标框定位，因此严格来说属于细粒度分类/识别，而不是目标检测。

## 方法

- 增广感知的 aHash + pHash 近重复检测
- Group-aware train/validation split，避免重复图像跨集合泄漏
- ResNet50、ConvNeXt-Tiny、ViT-B/16 主干网络
- 遥感旋转增强：水平/垂直翻转、90 度倍数旋转及小角度旋转
- Compact Bilinear Pooling 与 CBAM 消融实验
- Class-Balanced Cross Entropy 处理长尾类别
- AdamW、warmup、cosine learning-rate schedule 与 bfloat16 AMP

## 评测协议

- **分组感知三分切分**：近重复图像先被 `tools/dedup_check.py` 合并成组，整组进同一个 split，
  按类别分层，唯一图像组的 60% / 20% / 20% 进 train / val / test。
- **val 只负责选 epoch**（按全类 mean-per-class top-1），**test 在该 epoch 上只算一次**，报的就是它。
  没有独立测试集时，用同一个 val 既选 epoch 又报分等于取一条噪声序列的最大值，见 REPORT §4。
- **3 个种子**，种子同时控制切分与初始化，`mean ± std` 里含切分方差。
- 每档只比父级多改一个字段，`analyze.py` 自动打印 Δ 与父级字段 diff。

FGSCR-42 公开数据包含 7,778 张图像、42 个类别，去重后 5,233 个图像组。**FGSCR-42 没有官方
train/test 划分**，本仓库的数字不能与已发表结果直接比较。

## 当前结果

3 种子、test 在 val 选定 epoch 上的指标（mean ± std），完整表见 `runs/_analysis/TABLE.md`：

| 配置 | top-1 | MPC (support≥5) | 全类 MPC |
|---|---:|---:|---:|
| ResNet50 @224（L0） | 97.54 ± 0.15 | 95.66 ± 0.92 | 87.43 ± 1.37 |
| 同上，朴素切分（泄漏 39.78%） | 98.76 ± 0.13 | 99.08 ± 0.49 | 90.22 ± 2.20 |
| ResNet50 @448 + 旋转增广（L2） | 98.34 ± 0.43 | 97.11 ± 1.38 | 90.62 ± 2.31 |
| ConvNeXt-Tiny @448 + 旋转增广（L6a） | **99.09 ± 0.33** | 98.36 ± 0.82 | 92.12 ± 1.06 |
| ViT-B/16-384 @448 + 旋转增广（L6b） | 98.31 ± 1.23 | **98.58 ± 0.59** | **94.26 ± 0.77** |

主要发现：朴素切分抬高 top-1 约 1.2 点、support≥5 MPC 约 3.4 点；只用 val 的协议平均
高报全类 MPC 2 点；干净协议下多数模块差异落在种子噪声内，紧凑双线性明确有害，
朝向增广对有充分支撑的类别 +2 点但被它所需的缩放裁剪抵消。分析、局限与第一轮结论的
差异见 [REPORT.md](REPORT.md)。

## 环境

- Python 3.14
- PyTorch 2.11.0 + CUDA 12.8
- torchvision 0.26.0
- timm 1.0.27

安装核心依赖：

```powershell
python -m pip install -r requirements.txt
```

CUDA 版本的 PyTorch 建议根据显卡和 CUDA 环境从 PyTorch 官方安装入口选择对应命令。

## 数据准备

配置文件使用 ImageFolder 格式的数据目录（一个子目录一个类别）。所有命令在仓库根目录执行。

1. 数据体检与近重复检测，在**原始**图像上做，产出 `runs/_data/FGSCR-42/` 下的
   `report.md`、`stats.json`、`dup_groups.json`、`dedup_report.md`：

```powershell
python tools/inspect_data.py D:/path/to/FGSCR-42 --out runs/_data/FGSCR-42
python tools/dedup_check.py  D:/path/to/FGSCR-42 --out runs/_data/FGSCR-42 --val-ratio 0.2 --test-ratio 0.2
```

2. 建 JPEG 缓存。FGSCR-42 原包是 8.4 GB 的 BMP，60 个 epoch 直接读会被 I/O 主导；
   短边上限 512 高于阶梯里所有输入尺寸，没有实验会看到被降质的图：

```powershell
python tools/make_cache.py D:/path/to/FGSCR-42 D:/path/to/FGSCR-42-cache512
```

3. 在 `configs/fgscr42/L0.yaml` 里把路径改成你的，其余档位全部继承 L0：

```yaml
data:
  root: D:/path/to/FGSCR-42-cache512                 # 训练读缓存
  dup_groups: D:/path/to/fgvc/runs/_data/FGSCR-42/dup_groups.json   # 启用分组感知切分
```

## 训练与分析

先冒烟：每个 split 50 张、2 个 epoch，只求端到端跑通。

```powershell
python tools/train.py configs/fgscr42/L0.yaml --smoke
```

单个实验，或用 `--set` 覆写任意超参：

```powershell
python tools/train.py configs/fgscr42/L5.yaml
python tools/train.py configs/fgscr42/L5.yaml --set seed=1 train.epochs=30
```

整条实验阶梯。串行排队；已有 `metrics.json` 的档位自动跳过，留有 `latest.pt` 的从断点
精确恢复模型、优化器、学习率调度和随机状态：

```powershell
python tools/run_ladder.py configs/fgscr42/L*.yaml --seeds 0 1 2
```

汇总对照表、selection-gap 表、混淆矩阵和错分样例，写到 `runs/_analysis/`：

```powershell
python tools/analyze.py --runs runs --baseline L0
```

## 测试

纯 CPU 单元测试，覆盖切分的分组原子性、指标计算、CB-CE 加权、旋转增广无黑角、
配置继承和三种 head 的前向：

```powershell
python -m unittest discover -s tests -v
```

模型权重因文件较大未纳入 Git，仓库保留配置、日志、指标和可视化结果用于复现实验结论。
