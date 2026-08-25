# Experiment 1：多视图可靠性与拒绝实验记录

更新时间：2026-08-25  
当前阶段：IRSTD-1k Stage 1，mask 训练进行中

## 1. 固定配置

- generator：P0 选出的 neck spatial probe；
- views：identity、horizontal flip、vertical flip、local contrast、DoG/LoG-enhanced；
- 五个视图共享同一 encoder 与 probe 权重，通过 batch stacking 计算；
- 坐标逆变换后以 3 px 半径聚类，同一视图对一个 cluster 最多贡献一次；
- A2 score：`0.5 × mean_score + 0.5 × max_score`；
- A3 rule：`alpha=1.0, beta=0.5, gamma=1.0, max_dispersion=2 px, tau=0`；
- `min_support` 只在 validation 上比较预注册默认 3/5 与宽松 2/5；
- 没有文本、CLIP、CBGA、GT 点、GT 候选修正或强制补点。

## 2. Prompt-level 结果

False Prompts/MP、mean candidates 和 tiny recall 均按 K=20 报告。

| Method | Recall@5 | Precision@5 | Recall@20 | Tiny Recall@20 | False Prompts/MP | Mean K@20 | Candidate AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| A1 single view | 0.82906 | 0.25798 | 0.92308 | 0.84746 | 220.87 | 15.82 | 0.60140 |
| A2 five-view mean/max | 0.78632 | 0.23291 | 0.94017 | 0.89831 | 256.16 | 18.16 | 0.47348 |
| A3 default support≥3 | 0.85470 | 0.27100 | 0.92308 | 0.86441 | 165.94 | 12.22 | 0.65040 |
| **A3 validation-selected support≥2** | **0.85470** | **0.26385** | **0.96581** | **0.93220** | **223.35** | **16.05** | **0.61044** |

## 3. Validation gate 判断

相对 A2，`support≥2` 的 A3：

- False Prompts/MP 从 256.16 降到 223.35，下降 12.81%；
- overall Recall@20 从 0.94017 升到 0.96581，没有召回损失；
- tiny Recall@20 从 0.89831 升到 0.93220，且高于 A1 的 0.84746；
- Candidate AUPRC 从 A2 mean/max 的 0.47348 提高到 0.61044；
- 当前 validation 全部图像含目标，zero-prompt fraction 为 0，尚不能据此判断背景图拒绝安全性。

因此 A3 在 prompt-level 通过前三项定量闸门。默认 `support≥3` 虽将 False Prompts/MP 降低 35.22%，但相对 A2 的 Recall@20 下降 1.71 个百分点，超过 0.5pp 限制，所以不选。

该结论还不是最终 A3 通过：仍需等待 A1 100-epoch mask checkpoint，使用同一模型分别评估 A1/A2/A3 的 global IoU、mean nIoU、F1、Pd 与 Fa。只有最终 Fa/Pd 也符合条件，并且同一规则在 NUAA-SIRST validation 上方向一致，才允许训练 A4 ReliabilityHead。

## 4. DoG/LoG 提前诊断

在 learned probe 完成前，曾用当时最强无训练 generator DoG/LoG 跑低成本诊断。五视图 A2 相对单视图使 Recall@20 从 0.91453 降到 0.90598，tiny Recall@20 从 0.84746 降到 0.83051，False Prompts/MP 从 237.46 升到 266.65；几何三视图则与单视图逐项相同。这说明翻转一致性本身不能区分目标与稳定杂波，强度增强视图还可能破坏 tiny target。

DoG/LoG 不是最终选中的 A1 generator，因此这些结果只作为失败机制对照，不参与正式 A1–A3 主表。

## 5. 正在运行的 mask 实验

统一为 IRSTD-1k、100 epochs、seed 20260825、validation 固定 resize、前 60 epochs 冻结 encoder、HQ warm-up 30、固定 segmentation threshold 0.5：

| Run | Prompt input | 运行位置 | 状态 |
|---|---|---|---|
| A0 | null/no spatial prompt | local | 运行中 |
| A1-P | positive points, budget 5 | local | 运行中 |
| A1-D | dense targetness | server free GPU | 运行中 |
| A1-DP | dense targetness + positive points | local | 运行中 |

BF16 smoke test 已覆盖 P/D/DP；float16 首次 smoke 出现 non-finite loss，正式入口已改为 BF16，并在发现 non-finite loss 时立即返回非零错误。该异常只发生在被废弃的 smoke run，不进入结果表。

## 6. 停止条件

- 当前不启动 A4：mask-level A3 与 NUAA 方向一致性尚未完成；
- 当前不运行 NUDT-SIRST：它只用于最终确认，不参与 idea 筛选；
- 若 mask-level A3 的 Fa 未下降至少 10%，或 Pd/Component Recall 下降超过 0.5pp，则记录失败并停止 A4；
- 若 prompt 改善而最终 mask 不改善，则结论转为 decoder/prompt encoder 是瓶颈。

## 7. 可复现命令

```powershell
python scripts/eval_multiview_prompt_quality.py `
  --data_root <IRSTD-1k> `
  --split splits/experiment1_seed20260825/val.txt `
  --generator probe --probe_checkpoint <best_neck.pt> `
  --gate rule --min_support 2 --max_dispersion 2.0 `
  --output_dir outputs/experiment1_a3/IRSTD-1k/neck_rule_support2

python scripts/train_experiment1_single_view.py `
  --data_root <IRSTD-1k> --generator probe `
  --probe_checkpoint <best_neck.pt> `
  --prompt_input points --prompt_budget 5 `
  --epochs 100 --seed 20260825 --output_dir <run-dir>
```

