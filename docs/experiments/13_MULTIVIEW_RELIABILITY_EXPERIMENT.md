# Experiment 1：多视图可靠性与拒绝实验记录

更新时间：2026-08-26
当前阶段：IRSTD-1k Stage 1 已完成；A3 最终 mask 收益不成立，A4 暂停

## 1. 固定配置

- generator：P0 选出的 neck spatial probe；
- views：identity、horizontal flip、vertical flip、local contrast、DoG/LoG-enhanced；
- 五个视图共享同一 encoder 与 probe 权重，通过 batch stacking 计算；
- 坐标逆变换后以 3 px 半径聚类，同一视图对一个 cluster 最多贡献一次；
- A2 score：`0.5 × mean_score + 0.5 × max_score`；
- A3 从预注册默认 `alpha=1.0, beta=0.5, gamma=1.0` 起步；随后只用 validation 上缓存的 cluster 统计搜索 108 个低成本组合，不重复运行 encoder；
- 最终 validation-selected rule：`alpha=0.5, beta=0, gamma=0, min_support=2/5, max_dispersion=2 px, tau=0`；
- 没有文本、CLIP、CBGA、GT 点、GT 候选修正或强制补点。

## 2. Prompt-level 结果

False Prompts/MP、mean candidates 和 tiny recall 均按 K=20 报告。

| Method | Recall@5 | Precision@5 | Recall@20 | Tiny Recall@20 | False Prompts/MP | Mean K@20 | Candidate AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| A1 single view | 0.82906 | 0.25798 | 0.92308 | 0.84746 | 220.87 | 15.82 | 0.60140 |
| A2 five-view mean/max | 0.78632 | 0.23291 | 0.94017 | 0.89831 | 256.16 | 18.16 | 0.47348 |
| A3 default support≥3 | 0.85470 | 0.27100 | 0.92308 | 0.86441 | 165.94 | 12.22 | 0.65040 |
| A3 default support≥2 | 0.85470 | 0.26385 | 0.96581 | 0.93220 | 223.35 | 16.05 | 0.61044 |
| **A3 validation-selected rule** | **0.87179** | **0.26913** | **0.97436** | **0.94915** | **223.16** | **16.05** | **0.69916** |

## 3. Validation gate 判断

相对 A2，validation-selected A3：

- False Prompts/MP 从 256.16 降到 223.16，下降 12.88%；
- overall Recall@20 从 0.94017 升到 0.97436，没有召回损失；
- tiny Recall@20 从 0.89831 升到 0.94915，且高于 A1 的 0.84746；
- Candidate AUPRC 为 0.69916，高于 A2-max 的 0.61149 和 A2 mean/max 的 0.47348；
- 当前 validation 全部图像含目标，zero-prompt fraction 为 0，尚不能据此判断背景图拒绝安全性。

因此 A3 在 prompt-level 通过全部已可计算闸门。108 个组合中有 24 个同时满足虚警下降、overall/tiny recall 和优于 max-score AUPRC 的约束；按“先通过全部闸门，再选择最低 False Prompts/MP@20，随后比较 tiny/overall recall 与 AUPRC”的预注册顺序得到上述规则。默认 `support≥3` 虽将 False Prompts/MP 降低 35.22%，但相对 A2 的 Recall@20 下降 1.71 个百分点，超过 0.5pp 限制，所以不选。

规则分数与简单排序控制如下；它们都从同一份缓存 cluster 计算，不重复运行 encoder。

| Score | Candidate AUPRC | Recall@20 | Tiny Recall@20 | False Prompts/MP@20 |
|---|---:|---:|---:|---:|
| max score | 0.61149 | 0.95726 | 0.91525 | 255.78 |
| mean score | 0.45235 | 0.93162 | 0.88136 | 256.35 |
| support only | 0.35223 | 0.93162 | 0.86441 | 256.35 |
| **A3 selected rule** | **0.69916** | **0.97436** | **0.94915** | **223.16** |

掩码评测完成后，A3 相对 A2 的 prompt 条件仍成立，但这一改善没有转化为更低的最终 Fa 或更高的 global IoU，详见第 5 节。因此本节只能支持“规则改善候选排序/拒绝”，不能支持“规则改善最终分割”。此外，同一规则尚未在 NUAA-SIRST validation 上确认方向，A3 的完整六项闸门仍未通过。

## 4. DoG/LoG 提前诊断

在 learned probe 完成前，曾用当时最强无训练 generator DoG/LoG 跑低成本诊断。五视图 A2 相对单视图使 Recall@20 从 0.91453 降到 0.90598，tiny Recall@20 从 0.84746 降到 0.83051，False Prompts/MP 从 237.46 升到 266.65；几何三视图则与单视图逐项相同。这说明翻转一致性本身不能区分目标与稳定杂波，强度增强视图还可能破坏 tiny target。

DoG/LoG 不是最终选中的 A1 generator，因此这些结果只作为失败机制对照，不参与正式 A1–A3 主表。

## 5. 100-epoch mask 结果

统一为 IRSTD-1k、100 epochs、seed 20260825、validation 固定 resize、前 60 epochs 冻结 encoder、HQ warm-up 30、固定 segmentation threshold 0.5：

| Run | Prompt input | Best epoch | global IoU (%) | mean nIoU (%) | F1 (%) | Pd (%) | Fa (×10^-6) | latency (ms/image) | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| A0 | null/no spatial prompt | 82 | 56.02 | 47.86 | 60.34 | 88.89 | 50.54 | 26.88 | 完成 |
| A1-P | positive points, budget 5 | 84 | 55.61 | 48.13 | 60.20 | 88.03 | 44.06 | 25.10 | 完成；A2/A3 主线 checkpoint |
| A1-D | dense targetness | 88 | 55.62 | 49.52 | 61.37 | 86.32 | 39.67 | 25.07 | 完成 |
| A2 | five-view mean/max points | 84 | **56.09** | 48.01 | 60.11 | 86.32 | **33.57** | 106.25 | 完成；复用 A1-P checkpoint |
| A3 | five-view rule-gated points | 84 | 55.58 | **48.70** | **60.86** | **87.18** | 40.44 | 106.52 | 完成；复用 A1-P checkpoint |

这里的 latency 是同一次本地统一评测的端到端单图均值，只用于同表相对比较。A2/A3 约为 A1-P 的 4.23 倍，与五视图额外计算一致；不能把多视图计算收益表述为参数收益。

### 5.1 A1 prompt 组成选择

- A1-P 相对 A0：global IoU -0.42pp、Pd -0.85pp、Fa -12.83%；没有整体超过无 prompt 基线。
- A1-D 相对 A0：global IoU -0.40pp、Pd -2.56pp、Fa -21.51%，但 mean nIoU +1.66pp、F1 +1.03pp；同样是混合胜负。
- A1-DP 在本地完成 60/100 epochs 后，于第 61 epoch 发生 CUDA OOM；部分最佳为 epoch 52、global IoU 45.31%，不进入正式结果表，也不与完整 run 比较。
- 依预注册协议，A2–A4 优先使用可逐点拒绝的 A1-P。A1-D 没有显著更强，当前也没有逐样本 `dense_prompt_valid`；因此 A1-DP 即使完成也不能改变本轮主线选择。为避免无信息增益的重复计算，不重跑 A1-DP。

### 5.2 A2/A3 最终掩码比较

相对 A1-P，A2 的 global IoU +0.49pp、Fa -23.81%，但 Pd -1.71pp，mean nIoU/F1 也略降；它体现的是明显的低虚警—低检出权衡，而不是全面提升。相对 A0，A2 的 global IoU 仅 +0.07pp、mean nIoU +0.15pp、Fa -33.58%，但 Pd -2.56pp。

相对 A2，A3 的 mean nIoU +0.69pp、F1 +0.75pp、Pd +0.85pp，但 global IoU -0.52pp，Fa 从 33.57 增至 40.44（+20.45%）。因此 A3 的候选层收益没有转化为更好的最终 IoU/Fa；当前瓶颈更可能在 prompt budget、prompt encoder 或 decoder 对候选排序变化的响应，而不是候选可靠性分数本身。

### 5.3 Checkpoint 与评测证据

- A0 best-mask SHA-256：`722c7f7838aeca5517a54af6791662342aade8222392243a7db9111bfe39388b`；
- A1-P best-mask SHA-256：`6320c5e2a68aa934b92b869998d826463b630f560f96e4257391deebabc9a904`；
- A1-D best-mask SHA-256：`e94fefa22cd116f149f197544d90e5c9197dfa6f8ace9918d9717853a728ead8`；
- split SHA-256：`3f0206fda5f471690f47570f80990e81059e2282caa8739b04905879c29faa1e`；
- 训练输出：`outputs/experiment1_a0/IRSTD-1k/` 与 `outputs/experiment1_a1/IRSTD-1k/`；
- 统一评测输出：`outputs/experiment1_mask_eval/IRSTD-1k/`。

BF16 smoke test 已覆盖 P/D/DP；float16 首次 smoke 出现 non-finite loss，正式入口已改为 BF16，并在发现 non-finite loss 时立即返回非零错误。该异常只发生在被废弃的 smoke run，不进入结果表。

## 6. 闸门结论与停止决策

按原计划第 10.4 节逐项判断：

1. A3 相对 A2 的 False Prompts/MP 下降 12.88%，通过“False Prompts 或最终 Fa 至少下降 10%”中的 prompt 分支；最终 Fa 本身没有通过。
2. Component Recall@20 +3.42pp，Pd +0.85pp，没有下降，通过。
3. tiny Recall@20 为 94.92%，高于 A1 的 84.75%，通过。
4. zero-prompt fraction 为 0；当前验证集没有空目标图像，因此只确认未错误拒绝该验证集中的有目标图像，不能外推背景图安全性。
5. rule AUPRC 0.69916 高于 max-score 0.61149，通过。
6. NUAA-SIRST 同规则方向尚未验证，不通过。

因此，A3 只通过 IRSTD prompt-level 的五项可计算条件，完整闸门未通过。更重要的是，mask-level 没有形成稳定主指标收益。当前停止 A4、NUDT-SIRST 和多随机种子长训练；A3 保留为候选可靠性分析，不作为主模型。若后续继续，最小必要实验是先在 NUAA validation 复现同一固定规则，并检查“prompt 改善但 mask 不改善”是否仍然存在，而不是直接堆叠 ReliabilityHead。

## 7. 可复现命令

```powershell
python scripts/eval_multiview_prompt_quality.py `
  --data_root <IRSTD-1k> `
  --split splits/experiment1_seed20260825/val.txt `
  --generator probe --probe_checkpoint <best_neck.pt> `
  --gate rule --min_support 2 --max_dispersion 2.0 `
  --alpha 0.5 --beta 0 --gamma 0 `
  --output_dir outputs/experiment1_a3/IRSTD-1k/neck_rule_support2

python scripts/train_experiment1_single_view.py `
  --data_root <IRSTD-1k> --generator probe `
  --probe_checkpoint <best_neck.pt> `
  --prompt_input points --prompt_budget 5 `
  --epochs 100 --seed 20260825 --output_dir <run-dir>

python scripts/eval_experiment1_masks.py `
  --data_root <IRSTD-1k> `
  --split splits/experiment1_seed20260825/val.txt `
  --checkpoint <A1-P-best-mask.pt> `
  --probe_checkpoint <best_neck.pt> `
  --mode A3 --min_support 2 --max_dispersion 2 `
  --alpha 0.5 --beta 0 --gamma 0 `
  --output_dir outputs/experiment1_mask_eval/IRSTD-1k/A3_best_mask_seed20260825
```
