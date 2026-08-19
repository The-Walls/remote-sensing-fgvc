# 遥感细粒度识别 — 实现方案（历史规划）

状态：**seed0 全实验阶梯与分析已完成；待补 seeds 1/2 做统计复验。**
当前数据、实现和结果以 `REPORT.md` 与 `runs/_analysis/TABLE.md` 为准；下文保留为
立项时的历史方案，其中“FGSCR-42 未获取”等状态不再代表当前项目。
工作区：`D:\rs-project\fgvc`（C: 仅剩 6.4 GB，全部落 D:）
环境：Python 3.14.3 / torch 2.11.0+cu128 / timm 1.0.27 / RTX 5090 D，`venv` 在 `D:\rs-project\venv`

---

## 1. 数据现状

| 数据集 | 状态 | 图像数 | 类别 | 短边中位数 | 长尾比 |
|---|---|---:|---:|---:|---:|
| MTARSI-fixed | **已下载** `D:\rs-project\data\MTARSI-fixed` | 2264 | 27 | 189 px | 4.3x |
| FGSCR-42 | **未获取**，仅百度网盘 | ~9320 | 42 | 未知（50–1500 px） | 未知 |
| CIFAR-10 | 已在盘上（pickle） | 60000 | 10 | 32 px | 1.0x |

完整体检报告：`runs/_data/MTARSI-fixed/report.md`，图：`runs/_data/MTARSI-fixed/distribution.png`

---

## 2. 三个会直接影响结论的发现

### 发现 A：在 MTARSI 上，L1（@448）不是分辨率实验，是上采样实验

短边分布：p05=105 / p50=189 / p95=286，**最大值 360，100% 的图短边 < 448，71.8% < 224**。

从 224 升到 448 不会引入任何新像素信息，只是双线性插值放大 + 计算量翻 4 倍。
L1 若涨点，成因是「ViT/CNN 的有效感受野相对目标变小 + 计算量增加」，**不是**「保住了更多判别性像素」。
组会上按后者讲会被问倒。

- 在 MTARSI 上：把 L1 重新定义为 `224 → 320`（p95=286，320 基本是无损上界），并在 REPORT 里写明「本数据集原生分辨率不足以支撑 448，该级验证的是计算预算而非信息量」。
- 在 FGSCR-42 上：原图 50–1500 px，L1@448 才是**真正的**分辨率实验。这是必须拿到 FGSCR-42 的核心理由。

### 发现 B：MTARSI 的长尾是「截断」出来的，不是自然长尾

27 类里 20 类恰好 100 张 —— 数据集在 100 处被人为截断。真实尾部只有 4 类：
B-29(23) / E-2(24) / Su-34(26) / F-4(29)。长尾比 **4.3x**。

对比：主流长尾 benchmark 的不平衡比是 100x–256x。4.3x 属于**轻度不平衡**，
L5（重加权/平衡采样）在这个量级上的增益通常在噪声范围内，大概率跑出「没提升甚至掉点」。
这个结论本身可以讲（"轻度不平衡下重加权无收益"），但它不是你开题想要的那条证据。
FGSCR-42 的长尾要真实得多。

### 发现 C：验证集太小，单次运行的差异不可信

2264 张按 75/25 分层切分 → 训练 1698 / 验证 566。
- 单张验证图 = 0.18% overall acc
- 尾部类（B-29）验证集只有 **6 张** → 该类 per-class acc 的量化步长是 16.7%
- mean-per-class acc 由 27 类平均，尾部 4 类贡献 15% 权重却各自只有 6 张 → 方差极大

**后果**：L2 比 L1 高 1.5%，很可能只是随机种子。这种表格拿去组会，导师第一个问题就是
"跑了几次？"

**对策**：数据集小 = 训练快（1698 张 @224 在 5090 上约 3–5 s/epoch）。
每一级用 **3 个种子**跑（种子同时控制 split 和初始化），表里报 `mean ± std`。
7 级 × 3 种子 = 21 次运行，总耗时仍在 1 小时量级，远低于你 2 小时/实验的预算。

---

## 3. FGSCR-42 获取：需要你动手

唯一官方源是百度网盘，**必须你本人登录**，我不能代你登录账号：

- 链接：https://pan.baidu.com/s/1eXplDfB5fCBPm7WMcFKZkg
- 提取码：`9xx8`
- 出处：https://github.com/DYH666/FGSCR-42

我查过的替代源，全部无果：HuggingFace datasets（`FGSCR` / `FGSCR-42` / `fine-grained-ship` 三个查询均 0 命中）、
GitHub README（无 Google Drive / OneDrive 镜像）。

**请你下载后解压到 `D:\rs-project\data\FGSCR-42`**（保持 ImageFolder 结构：一个子目录一个类别）。
到位后我立刻用 `tools/inspect_data.py` 出同款体检报告，再决定 L1 的分辨率档位和 L5 的重加权强度。

你的 D: 盘剩 199 GB，放得下。

---

## 4. 实现方案

### 4.1 目录结构

```
D:\rs-project\fgvc\
├── configs\
│   ├── base.yaml                 # 所有默认超参 + 理由注释
│   ├── mtarsi\L0.yaml … L6b.yaml # 每级只覆写 1 个字段
│   └── fgscr42\L0.yaml … L6b.yaml
├── src\
│   ├── config.py       # yaml 加载 + 继承 + --set 命令行覆写 + 解析后完整落盘
│   ├── data.py         # ImageFolder + 分层切分 + 平衡采样器 + 长尾统计
│   ├── transforms.py   # 基础增广 / 遥感旋转增广（难点 2）
│   ├── models.py       # timm backbone + 可插拔 head（GAP / Bilinear / CBAM）
│   ├── heads.py        # 紧凑双线性池化（难点 3）、CBAM（难点 3）
│   ├── losses.py       # CE / class-balanced CE（难点 4）
│   ├── metrics.py      # top-1 / mean-per-class / macro-F1 / 混淆矩阵
│   └── engine.py       # 训练循环 + bf16 AMP + OOM 自动降 batch
├── tools\
│   ├── inspect_data.py # 已完成
│   ├── train.py        # 单次实验入口，--smoke 冒烟
│   ├── run_ladder.py   # 批量跑 L0–L6 × N 种子
│   └── analyze.py      # 扫 runs/ → 对照表 + 混淆矩阵 + 易混类对 + 错分拼图
├── runs\<exp>\seed<k>\ # metrics.json / config.yaml / log.txt / best.pt
│                       # confusion_matrix.png / curves.png / errors_grid.png
└── REPORT.md           # 每个模块对应哪条领域难点
```

### 4.2 实验阶梯（按你的表，两处按发现 A 调整）

| id | 相对父级改动 | 父级 | 验证假设 | 难点 |
|----|---|---|---|---|
| L0 | — | — | ResNet50 @224 基础增广，地板线 | — |
| L1 | `img_size: 224 → 320`（MTARSI）/ `→ 448`（FGSCR-42） | L0 | 分辨率 vs 目标像素少 | 1 |
| L2 | `aug: basic → rs_rot` | L1 | 旋转增广 vs 朝向任意 | 2 |
| L3 | `head: gap → compact_bilinear` | L2 | 二阶特征编码 | 3 |
| L4 | `head: gap → cbam` | L2 | 弱监督定位判别区域 | 3 |
| L5 | `loss: ce → cb_ce` 或 `sampler: none → balanced` | L2 | 长尾 | 4 |
| L6a | `model: resnet50 → convnext_tiny` | L2 | backbone 敏感性 | — |
| L6b | `model: resnet50 → vit_base_patch16` | L2 | backbone 敏感性 | — |

L3/L4/L5/L6 都从 L2 分叉，**与父级严格只差一个字段**。`analyze.py` 自动打印每行相对
L0 的 Δ 和相对父级的字段 diff。

### 4.3 默认超参（按细粒度分类常规做法定，写进 base.yaml）

| 项 | 值 | 理由 |
|---|---|---|
| optimizer | AdamW | 细粒度微调事实标准；ViT/ConvNeXt 必须用 |
| lr | 1e-4 | 全量微调预训练骨干的常规档；数据只有 ~1.7k 张，3e-4 会灾难性遗忘 |
| head lr multiplier | 10x | 新初始化的分类头需要更快收敛 |
| weight_decay | 0.05 | timm 对 ConvNeXt/ViT 的默认档 |
| scheduler | cosine，5 epoch warmup | 小数据集上 warmup 显著稳定前期 |
| epochs | 60 | 1698 张 / bs32 = 53 iter/epoch，60 epoch ≈ 3.2k iter，够收敛且不过拟合 |
| batch_size | 32 | @320 下 5090 显存充裕；OOM 时自动减半 |
| label_smoothing | 0.1 | 细粒度类间边界模糊，抑制过置信 |
| AMP | bf16 | Blackwell 原生支持，无需 GradScaler，比 fp16 稳 |
| 切分 | 分层 75/25，种子控制 | 见发现 C |
| 种子 | 0/1/2 三次 | 见发现 C，表里报 mean±std |

### 4.4 硬性要求对照

- ✅ backbone 全部走 `timm.create_model`，不手写
- ✅ `metrics.json` 存解析后的完整 config + git 状态 + 环境版本
- ✅ 记录 overall top-1 / mean-per-class top-1 / macro-F1 / 参数量 / 单 epoch 秒数
- ✅ `torch.amp` bf16；OOM 捕获 → batch 减半 → 日志显式写明 `[OOM] bs 32 -> 16`
- ✅ `analyze.py` 输出 markdown 对照表 + 每实验混淆矩阵 png + top-15 易混类对 + 12 张错分拼图（标注 GT/Pred）
- ✅ 所有中间产物落盘为 png，可直接拖进 PPT

---

## 5. 分阶段计划

| Stage | 内容 | 冒烟 | 停下来给你 review |
|---|---|---|---|
| 1 | `config.py` / `data.py` / `transforms.py` / `models.py` / `engine.py` / `train.py`，L0 跑通 | `--smoke`：每 split 50 张、2 epoch | L0 完整结果 + 曲线 |
| 2 | L1 / L2 完整跑通（×3 种子） | 同上 | L0/L1/L2 对照表 |
| 3 | `heads.py`（紧凑双线性、CBAM）+ `losses.py`（CB-CE），L3/L4/L5 | 同上 | L3/L4/L5 结果 |
| 4 | `analyze.py` + `REPORT.md` + L6a/L6b | 同上 | 全表 + 全部 png |

每个 Stage 开工前先跑 `--smoke`，端到端不报错才启动完整训练。

---

## 6. 需要你拍板

1. **主数据集**：先在 MTARSI-fixed 上把 L0–L6 全部跑通出表（今天就能有结果），
   FGSCR-42 到位后换 `data.root` 重跑一遍？还是只等 FGSCR-42？
2. **L5 的机制**：类别重加权 CE（class-balanced loss, Cui et al. 2019）还是平衡采样？
   我倾向 CB-CE —— 不改变数据管线，变量更干净。
3. **发现 A 的处理**：MTARSI 上 L1 用 320 而非 448，接受吗？
4. **3 种子**：接受 21 次运行换取 mean±std 吗？（总耗时仍 < 1 小时）
