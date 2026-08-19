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

当前 seed 0 最佳配置为 `L5`：

| Metric | Result |
|---|---:|
| Overall top-1 | 99.89% |
| Mean-per-class top-1 | 97.49% |
| MPC (support >= 5) | 99.82% |
| Macro-F1 | 97.37% |

完整实验分析见 [REPORT.md](REPORT.md)，组会说明见 `遥感图像细粒度识别_实现算法与组会汇报说明.docx`。

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

配置文件使用 ImageFolder 格式的数据目录。先在实验 YAML 中设置：

```yaml
data:
  root: D:/path/to/FGSCR-42
  dup_groups: runs/_data/FGSCR-42/dup_groups.json
```

执行数据审计与重复检测：

```powershell
python tools/inspect_data.py D:/path/to/FGSCR-42 --out runs/_data/FGSCR-42
python tools/dedup_check.py D:/path/to/FGSCR-42 --out runs/_data/FGSCR-42
```

## 训练与分析

运行单个实验：

```powershell
python tools/train.py configs/fgscr42/L5.yaml
```

运行实验阶梯：

```powershell
python tools/run_ladder.py --dataset fgscr42
```

汇总指标、混淆矩阵和错误样本：

```powershell
python tools/analyze.py --dataset fgscr42
```

模型权重因文件较大未纳入 Git，仓库保留配置、日志、指标和可视化结果用于复现实验结论。
