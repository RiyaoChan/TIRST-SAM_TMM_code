# Experiment 1：Self-Prompt 代码与历史协议审计

更新时间：2026-08-25  
分支：`codex/highres-multiview-self-prompt`

## 1. 审计结论

旧 EfficientSAM 工程中的结果可以帮助淘汰无效路线，但不能直接充当本实验的主结果。原因是旧实验普遍使用原始测试集选 checkpoint、至少强制输出一个候选点，或在训练阶段混入 GT 点；这些条件与 Experiment 1 的独立验证集、允许零候选、统一预算以及严格 image-only 推理协议不一致。

本轮因此采用以下最小重跑策略：

| 路线 | 历史结论 | 本轮处理 |
|---|---|---|
| PGAP | 三个数据集均有正结果，但旧协议不兼容 | 不重复训练生成器；在新验证集统一复评候选质量 |
| DoG/LoG | NUDT 轻微提升，IRSTD/NUAA 基本持平或下降 | 不重复旧式微调；在新验证集统一复评 |
| SelfPromptingHead | 有可用 checkpoint，但训练用 GT 做 hard-negative mining | 不复用主结果；训练严格空间 probe |
| two-stage self-prompt | 三个数据集均明显弱于对应 one-stage | 作为历史负结果归档，不再长训练 |
| DynamicSparsePrompt | 固定随机种子下与 baseline 基本相同或更差 | 属于无明确空间位置的 sparse token；归档并停止 |
| MultiLevelDynamicSparsePrompt | 与 baseline 基本相同或更差 | 同上；归档并停止 |

## 2. 代码身份核验

旧工程与当前工程的以下核心文件逐字节一致：

- `efficient_sam/PGAP.py`
- `efficient_sam/self_prompting_head.py`
- `sirst_dataset.py`

旧实验记录的本地副本与服务器副本 SHA-256 均为：

```text
81c5b2c39956a76ad0e0474090b3033798d71b36c0513148673abb24917e72bb
```

这说明历史结果与当前抽取模块之间具有可追溯关系。不过训练主入口已经发生变化，所以历史 checkpoint 仍需按协议逐项核查，不能仅凭模块哈希视为可直接比较。

## 3. GT 泄漏与训练—推理不一致

### 3.1 PGAP

PGAP 的相位显著图由图像直接计算，本身不需要 GT；但旧候选提取逻辑在没有峰值时会回退到全局最大值，并通过 `min_top_k` 补足点，因此无法表示“零 prompt”。此外，`label_points_by_gt` 会用 GT 给候选点标正负并补充正点，不能进入严格 image-only 推理路径。

本轮使用 `PGAPProposalAdapter` 直接读取显著图，再通过统一的局部极大值提取器输出候选，关闭强制回退并允许零候选。

### 3.2 SelfPromptingHead

旧通用 `SelfPromptingHead` 在训练时接收 `gt_mask` 做 hard-negative mining，而评估时没有 GT，存在训练—推理不一致。历史的 late-mix 方案还会在后期以一定概率使用 GT prompt，因此不能作为 image-only 主结果。

本轮 learned probe 只在 loss 计算中使用分割标注，proposal 接口只接受图像特征；验证和推理时 GT 仅在候选生成完成后交给指标模块。

### 3.3 DynamicSparsePrompt 系列

DynamicSparsePrompt 主要输出无显式坐标的 sparse token；MultiLevelDynamicSparsePrompt 虽带低分辨率 targetness 辅助图，但历史主路径仍不是统一的空间点候选接口。二者不能直接回答“哪里产生了 prompt、prompt 是否命中小目标”的问题，因此不纳入 A1–A4 主比较。

## 4. 历史结果与停止依据

以下数字仅用于路线筛选，不与新协议的结果合并报告。

### 4.1 two-stage self-prompt

| 数据集 | two-stage mIoU | 历史 one-stage mIoU | 判断 |
|---|---:|---:|---|
| NUDT-SIRST | 0.9177 | 0.9278 | 停止 |
| IRSTD-1k | 0.6602 | 0.7055 | 停止 |
| NUAA-SIRST | 0.7117 | 0.7539 | 停止 |

### 4.2 DynamicSparsePrompt / MultiLevelDynamicSparsePrompt

在历史固定种子实验中，baseline mIoU 为 0.7297；DynamicSparsePrompt 各变体约为 0.7287–0.7293，MultiLevelDynamicSparsePrompt 各变体约为 0.7288–0.7291，Pd 基本不变，Fa 也未出现稳定收益。按预先约定的资源控制原则，两条路线均停止长训练。

### 4.3 DoG/LoG

历史 60-epoch 微调相对对应基线的变化为：NUDT-SIRST `+0.0014`、IRSTD-1k `-0.0006`、NUAA-SIRST `-0.0012`。其主要现象是候选正点精度提高但组件覆盖下降，因此本轮只保留统一候选协议复评，不复制原训练。

### 4.4 PGAP

历史 PGAP 在 NUDT-SIRST、IRSTD-1k、NUAA-SIRST 上分别出现过约 0.9362、0.7747、0.7825 的 best mIoU，但这些实验使用原测试列表做验证，并设置 `min_top_k=1`，不能作为本轮主表结果。

## 5. 新协议的硬约束

1. 原始测试 ID 原样保留；从原训练集按目标存在、组件数、最大组件面积和总面积分层划出 10% 验证集。
2. 所有阈值和 checkpoint 只在验证集选择；冻结配置后才运行测试集。
3. 候选统一为 `(x, y, score, valid)`，`K_raw=32`，NMS 半径 3 px，并报告预算 `K={1,3,5,10,20,32}`。
4. proposal 生成函数不接受 mask、GT 点或文本；空显著图必须允许输出零候选。
5. GT 只允许用于训练 loss 和 proposal 完成后的离线指标计算。
6. learned probe 的 checkpoint 按 tiny component Recall@20、overall Recall@20、较低 False Prompts/MP、Dense AUPRC 的顺序在验证集选择。

## 6. P0 首轮无训练候选复评（IRSTD-1k validation）

固定 resize、`K_raw=32`、NMS 3 px、score threshold 0.10，共评估 80 张验证图像、117 个 resize 后连通域。

| 方法 | Candidate AUPRC | Dense AUPRC | Recall@5 | Recall@20 | Recall@32 | Tiny Recall@20 | False prompts/MP@20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| PGAP | 0.6213 | 0.1451 | 0.6496 | 0.7009 | 0.7094 | 0.6271 | 149.35 |
| DoG/LoG | 0.4893 | 0.1581 | 0.8120 | 0.9145 | 0.9231 | 0.8475 | 237.47 |

解释：DoG/LoG 在相同预算下明显提高小目标组件覆盖，但产生更多误候选；PGAP 候选分数排序更好、误候选较少。learned early/mid/neck probe 完成后，再决定 A1 使用哪个单视图生成器。

## 7. 可复现入口

```powershell
python scripts/build_experiment1_splits.py --help
python scripts/eval_prompt_quality.py --help
python scripts/train_prompt_probes.py --help
python -m pytest tests/test_no_gt_prompt_leakage.py tests/test_prompt_proposal.py -q
```

原始运行产物保存在忽略版本控制的 `outputs/experiment1_p0/`；后续将把冻结配置、摘要指标和停止/继续决策写入版本化实验记录。
