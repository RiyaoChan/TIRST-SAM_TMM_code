# Experiment 1：Prompt Generator 筛选结果

执行日期：2026-08-25  
数据集：IRSTD-1k 新 validation split（80 张，固定 resize 256×256）  
随机种子：20260825

## 1. 协议

- 原始 test split 完全保留；从原训练集按目标存在、组件数、最大组件面积和总前景面积分层划出 720 train / 80 validation；
- `K_raw=32`，局部极大值 NMS 半径 3 px，score threshold 0.10；
- 候选与 GT 的匹配采用按分数排序的确定性一对一 greedy matching；
- 命中条件为落入组件 2 px 膨胀区域，或距组件质心不超过 3 px；
- learned probe 冻结 EfficientSAM-Ti encoder，只训练轻量空间 head 20 epochs；
- checkpoint 排序依次使用 tiny Recall@20、overall Recall@20、较低 False Prompts/MP、Dense AUPRC；
- GT 只构造训练 loss，并在候选完全生成后进入指标模块；proposal sampler 不接收 mask。

## 2. 完整 P0 结果

False Prompts/MP 按 K=20 报告。

| Method | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Recall@20 | Tiny Recall@20 | Precision@5 | False Prompts/MP | Dense AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PGAP | 0.45299 | 0.64103 | 0.64957 | 0.68376 | 0.70085 | 0.62712 | 0.25166 | 149.35 | 0.14514 |
| DoG/LoG | 0.46154 | 0.76068 | 0.81197 | 0.84615 | 0.91453 | 0.84746 | 0.25000 | 237.46 | 0.15814 |
| Early probe | 0.45299 | 0.62393 | 0.72650 | 0.78632 | 0.88889 | 0.77966 | 0.21519 | 243.95 | 0.19783 |
| Mid probe | 0.51282 | 0.75214 | 0.80342 | 0.83761 | 0.89744 | 0.79661 | 0.26039 | 171.28 | 0.35704 |
| **Neck probe** | **0.48718** | **0.77778** | **0.82906** | **0.87179** | **0.92308** | **0.84746** | **0.25798** | **220.87** | **0.41523** |

## 3. 选择决定

Neck probe 被选为 A1 generator：

1. tiny Recall@20 与 DoG/LoG 并列最佳，均为 0.84746；
2. overall Recall@20 为 0.92308，高于 DoG/LoG 的 0.91453；
3. False Prompts/MP 为 220.87，低于 DoG/LoG 的 237.46；
4. Dense AUPRC 为 0.41523，明显高于全部无训练生成器；
5. Recall@5 为 0.82906，也是五种方法中最高。

因此不触发新的 NativeResolutionPromptHead：现有 neck probe 已满足 tiny/overall 召回容差，且没有证据表明 early feature 优于 neck feature。新增 full-resolution 分支只会引入尚未被证据支持的容量。

## 4. 历史路线的复用/停止

- PGAP、DoG/LoG：保留为 P0 无训练基线，不复制旧的 test-as-validation 微调；
- two-stage self-prompt：历史三个数据集均弱于 one-stage，停止；
- DynamicSparsePrompt、MultiLevelDynamicSparsePrompt：历史固定种子结果与 baseline 持平或更差，且主输出为无坐标 token，停止；
- 旧 SelfPromptingHead：训练使用 GT hard-negative mining，不复用为严格 image-only 主结果；
- 新 learned probe：训练与推理使用同一个候选提取器，验证 sampler 不接收 GT。

## 5. Checkpoint 与可复现性

三个最佳 probe 分别来自：early epoch 13、mid epoch 19、neck epoch 15。原始 checkpoint 和完整 CSV 位于忽略版本控制的 `outputs/experiment1_p0/IRSTD-1k/`。

同一 neck checkpoint 连续运行三次固定 validation 评测，四类核心产物的 SHA-256 均逐字节一致：

| Artifact | 三次共同 SHA-256 |
|---|---|
| `candidate_budget_curve.csv` | `fa75c3f7e839f47d287540291ca3acd51d28af9710c2cc8b4df6c827b24b7fc4` |
| `area_bin_metrics.csv` | `df62d56ebedaa27146572905de43f86a98c3e44ec585b3992ca9f1e502f28d84` |
| `prompt_metrics_per_image.csv` | `5eaed6802d712d1e801110b9f73ba3c3841415b79ea42548c7e65967313a4456` |
| `prompt_metrics_per_component.csv` | `5895513a45b8ac4b5b74787a75f9f61ad5e38a6365e2b7ca6e4500e138c38015` |

```powershell
python scripts/train_prompt_probes.py `
  --data_root <IRSTD-1k> `
  --output_dir outputs/experiment1_p0/IRSTD-1k/probe_20ep_seed20260825 `
  --epochs 20 --batch_size 4 --seed 20260825

python scripts/eval_prompt_quality.py `
  --data_root <IRSTD-1k> `
  --split splits/experiment1_seed20260825/val.txt `
  --generator probe `
  --probe_checkpoint <best_neck.pt> `
  --output_dir outputs/experiment1_p0/IRSTD-1k/probe_neck_best
```

本表只报告 validation 机制筛选结果，尚未运行 test；不得与旧 test-as-validation 数字直接合并。
