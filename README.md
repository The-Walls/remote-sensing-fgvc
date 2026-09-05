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

## 当前结果

FGSCR-42 公开数据包含 7,778 张图像、42 个类别。去重后得到 5,233 个图像组，朴素随机切分会使 44.78% 的验证图像在训练集中存在近重复样本。

指标为**收敛值**（最后 10 个 epoch 的均值 ± epoch 间抖动），不是验证集上最好的那个 epoch——本项目没有独立测试集，用同一个 val 既选 checkpoint 又报分数会奖励曲线抖动最大的配置，理由与实测见 [REPORT.md](REPORT.md) §4.0。

当前 seed 0 最佳配置为 `L5`（ResNet50 @448 + 旋转增强 + class-balanced CE）：

| Metric | Result |
|---|---:|
| Overall top-1 | 99.82 ± 0.05 |
| Mean-per-class top-1 | 97.47 ± 0.02 |
| MPC (support >= 5) | 99.79 ± 0.02 |
| Macro-F1 | 97.21 ± 0.09 |

但 `L2`/`L5`/`L6a`/`L6b` 在 30 个 support≥5 的类别上全部落在 99.71–99.79%，彼此无差异；模块之间的差距只出现在验证支持 1–4 张的尾部类，单种子下属候选信号。完整实验分析见 [REPORT.md](REPORT.md)，组会说明见 `遥感图像细粒度识别_实现算法与组会汇报说明.docx`。

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
python tools/dedup_check.py  D:/path/to/FGSCR-42 --out runs/_data/FGSCR-42
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
  native_short_side_p95: 1112                        # inspect_data.py 对原图的统计
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
python tools/run_ladder.py configs/fgscr42/L*.yaml --seeds 0
python tools/run_ladder.py configs/fgscr42/L*.yaml --seeds 1 2   # 多种子复验
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
