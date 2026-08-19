# 遥感图像细粒度识别 — 实验报告

> ## ⚠️ 数据泄漏警告（置顶，按阈值触发）
>
> **FGSCR-42 的朴素分层切分会造成 44.78% 的验证集泄漏。**
>
> 全库 7778 张图中，**3703 张（47.6%）属于某个近重复组**；按 aHash+pHash 双哈希、
> 叠加 rot90/180/270 与水平/垂直翻转变体、全图与中心裁剪双尺度、汉明距离 ≤5 判定。
> 去重后唯一图像只剩 **5233** 张（塌缩 32.7%）。若不做分组感知切分，1943 张验证图中
> 有 **870 张**在训练集里存在孪生，验证指标会混入对近重复样本的记忆。
>
> 本报告所有结果均使用**分组感知切分**（同一重复组整组进同一 split，泄漏归零）。
> `L0` 与 `L0_naive_split` 是专门用于量化该效应的对照，两者除
> `data.group_aware_split` 外配置完全一致。实测朴素切分相对 L0 的 overall top-1
> **-0.16** 个百分点、全类 mean-per-class **+1.96** 个百分点，说明泄漏确实改变
> 结论，但不会简单地把所有指标一并抬高。

---

## 1. 数据集审计

### 1.1 FGSCR-42（主数据集）

来源：[DYH666/FGSCR-42](https://github.com/DYH666/FGSCR-42)（百度网盘发布）。
公开发布版实际到手 **7778** 张，少于论文标称的 ~9320 张。
原始格式为未压缩 BMP，共 8.4 GB；已用 `tools/make_cache.py` 转为短边 ≤512 的 JPEG
缓存（663 MB，7.5%），512 高于所有分辨率扫描档位，实验不会看到被降质的图像。
解压包内的 macOS 残留（`__MACOSX` 下 6275 个 `._*` 文件、23 个 `.DS_Store`）已删除。

| 指标 | 值 |
|---|---:|
| 类别数 | 42 |
| 图像数（原始 / 去重后唯一） | 7778 / **5233** |
| 每类中位数 | 78 |
| 最多类 `029.Towing_vessel` | 778 |
| 最少类 `037.Horizon-class_destroyer` | **2** |
| 长尾比（原始 / 去重后 / 训练集实际） | 389x / 274x / **583x** |
| < 30 张的类 | 14 |
| 短边 p05 / p50 / p95 | 64 / 512 / 1112 |
| 短边 < 224 的图像 | 1884（24.2%） |
| 短边 < 448 的图像 | 3404（43.8%） |

图：`runs/_data/FGSCR-42/distribution.png`、`dup_examples.png`、`dup_groups.png`

**分辨率扫描在本数据集上是有效实验。** 短边中位数 512、p95 1112，224→320→448
每一档都在读取真实像素，而非插值。同时「24.2% 的图像短边不足 224」本身即是
领域难点 1（目标像素少）的量化证据。

**尾部类的统计约束。** 去重后 `037.Horizon-class_destroyer` 仅 2 张、
`035.Zumwalt-class_destroyer` 3 张，7 个类 < 10 张，14 个类 < 20 张。
处理方式：**保留全部 42 类**以保持与已发表结果的可比性，保证每类至少 1 张验证样本；
`metrics.json` 同时记录全 42 类的 mean-per-class 与**仅验证支持 ≥5 的类**的
mean-per-class（`mean_per_class_top1_support_ge5`），两者在对照表中并列，
使尾部噪声可被隔离而非被平均掉。

### 1.2 MTARSI-fixed（框架验证 / 数据质量审计）

来源：[amistele/MTARSI-fixed](https://huggingface.co/datasets/amistele/MTARSI-fixed)，
原始 MTARSI 见 [Zenodo](https://zenodo.org/records/3464319)。2264 张 / 27 类。

该数据集**不适合承担本课题的核心结论**，原因有三，均由体检得出：

1. **分辨率不足**：短边 p95 = 286，最大 360，100% 的图短边 < 448。
   在其上做 @448 实验只是上采样，不构成分辨率假设的检验。
2. **长尾是截断出来的**：27 类中 20 类恰好 100 张，说明在 100 处被人为砍平；
   真实尾部仅 4 类，长尾比 4.3x，属轻度不平衡，重加权方法在此量级增益通常落在噪声内。
3. **图像为合成**：检出 63 对「共享背景板但目标不同」的图像（100 张，4.4%），
   其中 6 对跨类——同一块停机坪上分别贴着 C-130 和 F-16。
   这意味着**领域难点 3（背景强干扰）在该数据集上不成立**，背景是复用的板子而非真实场景杂波。
   图：`runs/_data/MTARSI-fixed/dup_crossclass.png`

另检出真重复 226 对 / 371 张（16.4%），朴素切分泄漏率 13.81%。

---

## 2. 模块 ↔ 领域难点对应

| 模块 | 文件 | 对应领域难点 |
|---|---|---|
| 分辨率作为一等配置变量 + 每档标注是否超出原生分辨率 | `configs/*/L1_*.yaml`, `train.py` | **难点 1** 目标像素少 |
| `rs_rot` 增广：二面体群（h/v 翻转 + 90/180/270）+ ±30° 自由旋转，先放大 1.15x 再中心裁剪以消除旋转黑角 | `src/transforms.py` | **难点 2** 朝向任意 |
| `CompactBilinearPooling`：Tensor Sketch 近似二阶特征编码，替代会被背景稀释的全局平均池化 | `src/models.py` | **难点 3** 背景强干扰 |
| `CBAM`：通道 + 空间注意力，池化前对判别区域重加权（弱监督定位） | `src/models.py` | **难点 3** 背景强干扰 |
| `cb_ce`：类别平衡交叉熵（Cui et al. 2019），按有效样本数 `(1-β^n)/(1-β)` 重加权，β=0.9999；先计算 label-smoothed CE，再仅按真实标签权重加权样本 | `src/losses.py` | **难点 4** 长尾 |
| `mean_per_class_top1` / `macro_f1` / `..._support_ge5` 与 overall 并列上报 | `src/metrics.py` | **难点 4** 长尾 |
| 增广感知去重 + 分组感知切分（旋转/翻转变体哈希，双尺度判据） | `tools/dedup_check.py`, `src/data.py` | **难点 2 + 4 的交叉后果**：数据集用旋转副本扩充稀有类，直接制造验证集泄漏 |

---

## 3. 实验阶梯

每一级相对其父级**只改一个字段**；`analyze.py` 自动打印该字段 diff 与相对 L0 的 Δ。

| id | 父级 | 改动 | 验证假设 |
|---|---|---|---|
| L0 | — | ResNet50 @224 基础增广 | 地板线 |
| L0_naive_split | L0 | `group_aware_split: true→false` | **量化数据泄漏的贡献** |
| L1_320 / L1_448 | L0 | `img_size: 224→320/448` | 分辨率 vs 目标像素少 |
| L2 | L1_448 | `aug: basic→rs_rot` | 旋转增广 vs 朝向任意 |
| L3 | L2 | `head: gap→compact_bilinear` | 二阶特征编码 |
| L4 | L2 | `head: gap→cbam` | 弱监督定位 |
| L5 | L2 | `loss: ce→cb_ce` | 长尾（重点看 mean-per-class） |
| L6a / L6b | L2 | `model: resnet50→convnext_tiny / vit_base_patch16` | backbone 敏感性 |

---

## 4. 结果

见 `runs/_analysis/TABLE.md`（由 `tools/analyze.py` 自动生成）。

当前表为 **seed 0 单次运行**，因此小于约 1 个百分点的差异只能视为候选信号，不能替代
多种子 `mean ± std`。所有 checkpoint 均按全类 mean-per-class 选择。

### 4.1 基线、泄漏与分辨率

- `L0`：**overall 98.62% / 全类 MPC 90.27% / support≥5 MPC 98.04% /
  macro-F1 89.88%**（epoch 42）。overall 与全类 MPC 相差 8.35 点，直接显示 583x
  长尾下只报 overall 会掩盖尾类失败。
- `L0_naive_split` 相对 L0：overall **-0.16** 点、全类 MPC **+1.96** 点、
  support≥5 MPC **+0.25** 点、macro-F1 **+1.72** 点。44.78% 泄漏客观存在并主要
  影响尾类汇总，但没有把 overall 简单抬高；不得声称公开高分完全由泄漏支撑。
- `L1_320` 相对 L0：overall **+0.69** 点、全类 MPC **+2.42** 点、macro-F1
  **+3.31** 点，说明增加有效输入像素有收益。
- `L1_448` 相对 L1_320：overall **-0.21** 点、全类 MPC仅 **+0.12** 点、
  macro-F1 **-0.62** 点，而每 epoch 从 14.0 s 增至 26.0 s。单种子下 448 不具备
  性价比优势；后续多种子若无反转，主配方应优先采用 320。

### 4.2 模块消融

- `L2`（旋转增强）相对 `L1_448`：overall **+0.69** 点、全类 MPC **+4.65** 点、
  support≥5 MPC **+1.51** 点、macro-F1 **+4.74** 点。它是本阶梯中最明确、最符合
  遥感朝向任意先验的有效模块。
- `L3`（紧凑双线性）相对 `L2`：overall **-0.64** 点、全类 MPC **-5.84** 点、
  macro-F1 **-6.04** 点，且每 epoch 更慢（31.1 s vs 26.0 s），明确不保留。
- `L4`（CBAM）相对 `L2`：overall **-0.05** 点、全类 MPC **-1.19** 点、
  support≥5 MPC 基本相同、macro-F1 **-0.81** 点；没有证据支持额外注意力模块。
- `L5`（CB-CE）取得 seed0 最佳综合值：**99.89 / 97.49 / 99.82 / 97.37**
  （依次为 overall / 全类 MPC / support≥5 MPC / macro-F1）。相对 `L2` 的提升仅
  **+0.11 / +0.03 / +0.04 / +0.06** 点，必须经 seeds 1/2 才能判定是否真实。

`cb_ce` 曾发现并修复一个关键实现陷阱：直接把 `weight` 与 `label_smoothing=0.1`
同时传给 PyTorch `CrossEntropyLoss`，会把最高达 12.80 的稀有类权重施加到平滑目标的
所有类别项，首轮 top-1 仅 0.05%。当前实现先计算逐样本平滑 CE，再按真实标签对应权重
归一化加权；数值测试证明 label smoothing=0 时与标准加权 CE 等价（误差 1.19e-7）。

### 4.3 主干敏感性与尾类统计

- `L6a ConvNeXt-Tiny`：**99.89 / 97.49 / 99.82 / 97.35**，与 L5 实质并列；
  参数从 23.59M 增至 27.85M，seed0 没有足够证据证明值得替换 ResNet50。
- `L6b ViT-B/16`：前三项同为 **99.89 / 97.49 / 99.82**，但 macro-F1 只有
  **96.58**，参数 86.28M、每 epoch 50.4 s。它能打平类别召回峰值，却有更多误报且
  计算成本最高，不作为首选。

验证支持仅 1 张的类别每多判对 1 张，全类 MPC 就跳 **1/42 = 2.38** 个百分点；
支持 2 张的类别单张跳变为 **1.19** 点。因此 `L2/L4/L5/L6` 的 96–97% 峰值必须和
`support≥5` MPC 并看。对 30 个支持≥5的类别，强方案仅分布在 **99.78–99.82%**，
差异不足 0.05 点；当前最稳健结论是“旋转增强有效”，而不是“某个强主干绝对获胜”。

---

## 5. 复现

```bash
python tools/inspect_data.py  <root> --out runs/_data/<name>
python tools/dedup_check.py   <root> --out runs/_data/<name>
python tools/make_cache.py    <root> <cache> --max-short 512
python tools/run_ladder.py    configs/fgscr42/*.yaml --seeds 0
python tools/analyze.py       --runs runs --baseline L0
```

正式统计结论需补跑 `--seeds 1 2`；批量脚本会跳过 seed0，并从每个未完成任务的
`latest.pt` 精确恢复模型、优化器、学习率调度器和随机状态。

环境：Python 3.14.3 / torch 2.11.0+cu128 / timm 1.0.27 / RTX 5090 D。
`cudnn.deterministic = True`（未为速度放弃）。混合精度使用 **bfloat16**
（Blackwell 原生支持，无需 GradScaler）。

### 已知工程约束

- Windows 的 DataLoader 共享内存由 C: 分页文件（4 GB）支撑。8 workers × 448px ×
  并发 3 个任务会触发 `Couldn't open shared file mapping ... 1455`。
  处置：`data.workers` 降至 4，`run_ladder.py` 串行排队。分页文件未做改动
  （C: 仅剩 6.4 GB，且本机硬约束为 C: 不放占空间的东西）。
- 训练一律读 JPEG 缓存而非原始 BMP，否则 60 epoch × 8.4 GB 的 I/O 会主导耗时。
- 每个 epoch 原子写入 `latest.pt`，最佳模型单独写入 `best.pt`。前者用于严格续训，
  后者用于分析，避免中断后用“最佳权重 + 重置优化器”的非等价近似恢复。
