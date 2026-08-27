# MicroQuery M2-S1b：Component-Safe Rejection 实验记录

日期：2026-08-27

状态：**按门禁停止（未进入 hard-negative、第二数据集或长训练）**

## 1. 实验目的与边界

本实验检验：在不改变冻结候选、SAM query mask、A1-P encoder 和 probe 的前提下，能否同时做到：

1. 不丢失已被 K=10 候选覆盖的 GT component；
2. 拒绝至少 70% 的背景候选；
3. 将 Fa 控制在 `48.24×10^-6` 以内；
4. 不降低 Pd，并相对 one-query 改善 IoU 或 nIoU。

可部署 forward 只接收 ROI descriptor 和 candidate validity；分组只使用候选坐标、query probability、descriptor 与有效位。GT component membership 和 query IoU 只用于训练目标和验证统计，不进入推理。

## 2. 冻结证据

| 证据 | SHA-256 |
|---|---|
| K=10 candidate cache | `4269eb06c7f7d7dcec1edc55cdf4676bca4da700e95c58ad2963db5451766230` |
| validation feature/query-mask cache | `37ed8fdaac4ea59cd3cfaefd60c5d38ae42617366770afc4a7e12d9bcf99e26a` |
| validation analysis targets | `aefcb017d515caed3e479f27791b5338fd86c368e57cfe2dbe5589461de0e5e3` |
| train feature cache | `8964a228987ed874b1549428d01f6337593f2cf36dbc4febd2c78b6a1684d8cd` |
| train analysis targets | `0ae3274c22ea147b8d39adee393065291ac7c8c9a9e8209c2d3705401e26cce8` |
| old M2 objectness checkpoint | `0e9ddc5b2fdaf5a539ac54285e044ec23b3d7d9058d419f20ce40846ef949dac` |
| A1-P best-mask checkpoint | `6320c5e2a68aa934b92b869998d826463b630f560f96e4257391deebabc9a904` |
| probe checkpoint | `cc96c90cd19f4b215c535656055e5788b5d9db5cbe2f1fbc2910bb87faa0ae47` |
| validation split | `3f0206fda5f471690f47570f80990e81059e2282caa8739b04905879c29faa1e` |

冻结代码基线 commit：`3b9a75b7a5412c5951741b0e03828164129dc7bf`。验证集 80 张、117 个 GT component，其中 112 个被 K=10 候选覆盖。

## 3. A0 分组审计

对 coordinate-only、mask-only、hybrid 81 组网格和 feature-control 做了确定性审计。最终选择：

```text
G2_rn2_rf8_iou0.2_rm5
r_near=2, r_far=8, SoftIoU=0.2, mask-centroid radius=5
```

| 指标 | 结果 |
|---|---:|
| groups | 328（4.10/image） |
| collision groups | **0** |
| target groups | 116 |
| background groups | 212 |
| target pair grouping rate | 38.46% |
| duplicate-primary grouping rate | 55.56% |
| target-group background contamination | 0% |

结论：G2 满足“不同 GT component 不得被合并”的硬安全边界，但只合并了部分重复候选，因此分组自身的可获收益有限。

## 4. A1–A5 缓存策略

比较了 current hard gate、Top-L、group champion、tri-state rejection，以及 objectness/raw score/SAM quality 的组内 champion 组合。没有配置通过完整门禁。

| 条件 | FLCC | FCRR | CTR | IoU | nIoU | Pd | Fa (`×10^-6`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| M2 independent all | 0 | 0.00% | 92.86% | 50.34% | 46.79% | 89.74% | 90.98 |
| old hard gate (`τ=0.15`) | 5 | 83.86% | 92.86% | 54.37% | 50.25% | 88.89% | 43.87 |
| A2 Top-3（词典序最安全） | **0** | 59.19% | 92.86% | 51.65% | 48.17% | 89.74% | 75.15 |

Top-3 保住了所有 covered component，但背景拒绝和最终 Fa 均失败。分组型 A3/A4 在达到约 83.86% FCRR 时仍为 FLCC=5，没有修复旧 hard gate 的组件丢失。

反事实也表明 Top-3 主要是“多留候选”的数量保护：其 Correct、Random-groups、Coordinate-groups 和 Mask-groups 的 mask 结果相同，不能据此声称 component grouping 带来有效机制增益。

## 5. B1/B2：semantic 标签修正

标签改为：

```text
primary   -> target-like
duplicate -> target-like
background -> background
```

头部为 451→256→256 的共享 MLP，训练参数 183,177，小于 0.5M；20 epochs、seed 20260825、AdamW、lr `3e-4`、batch size 64，encoder/probe/SAM 全冻结。

训练后期 semantic AUPRC 最高达到 **0.9385**（raw candidate 为 0.8253），说明标签拆分本身有效。但 checkpoint 必须按 `FLCC→TCR→FCRR→Fa→CTR→nIoU` 选择，不能按 AUPRC 取后期模型。

| B1/B2 观察点 | FLCC | FCRR | IoU | nIoU | Pd | Fa (`×10^-6`) |
|---|---:|---:|---:|---:|---:|---:|
| 正式 `best_component_safe.pt`（epoch 2, τ=0.04） | **0** | 24.22% | 50.40% | 46.85% | 89.74% | 90.22 |
| FCRR≥70% 的最佳 epoch 观察点（epoch 13） | 4 | 78.48% | 53.86% | 49.55% | 88.89% | 48.45 |

结论：semantic 表示可以分离背景，但仍不能在相同阈值下同时保住组件。

## 6. B3：增加 utility regression + pairwise ranking

增加 query-IoU SmoothL1 和组内 representative ranking，未加 coverage loss。

| B3 观察点 | FLCC | FCRR | Fa (`×10^-6`) |
|---|---:|---:|---:|
| 正式 checkpoint（epoch 1, τ=0.01） | **0** | 4.93% | 90.98 |
| FCRR≥70% 的最佳 epoch 观察点（epoch 13） | 4 | 78.03% | 48.83 |

utility 将 duplicate suppression 的中期值提高到约 66.67%，但 group survival 仍由 semantic confidence 决定，因此没有解决 FLCC–FCRR 冲突。

## 7. B4：增加 coverage-safe loss

使用计划冻结权重：`λq=0.5, λr=0.2, λc=0.5, λb=0.1`。

| B4 观察点 | FLCC | FCRR | IoU | nIoU | Pd | Fa (`×10^-6`) |
|---|---:|---:|---:|---:|---:|---:|
| 正式 checkpoint（epoch 1, τ=0.01） | **0** | 4.93% | 50.34% | 46.79% | 89.74% | 90.98 |
| FLCC≤1 中 FCRR 最高的 epoch 观察点（epoch 12） | **1** | 56.95% | 51.04% | 47.46% | 89.74% | 81.06 |
| FCRR≥70% 的最低-FLCC观察点（epoch 15） | 4 | 71.30% | 53.02% | 48.38% | 88.89% | 57.60 |
| 最低 Fa 观察点（epoch 18） | 5 | 81.61% | 54.19% | 50.23% | 88.89% | 45.97 |

coverage loss 的确改善了低阈值下的 component survival，但没有让 FLCC≤1 与 FCRR≥70%、Fa≤48.24×10^-6 同时成立。B4 未达到计划规定的成功条件。

## 8. 最终门禁结论

| 门禁 | 要求 | 是否存在同时满足的配置 |
|---|---:|---:|
| FLCC | ≤1 | 单独可满足 |
| CTR drop | ≤0.5pp | 单独可满足 |
| FCRR | ≥70% | 单独可满足 |
| Fa | ≤48.24×10^-6 | 单独可满足 |
| Pd | ≥88.89% | 单独可满足 |
| one-query IoU/nIoU 增益 | 主指标 +0.5pp，另一项不降>0.5pp | 部分点可满足 |
| **全部门禁交集** | 同时满足 | **不存在** |

因此执行计划的停止规则：

- 不构建/评估 hard-negative split；
- 不运行 NUAA-SIRST 第二验证集；
- 不运行 100 epochs 或三种子；
- 不进入 test；
- 不把 M2-S1b 写成有效改进。

输出目录中保存了 `SKIPPED.json`，明确记录 hard-negative 与第二数据集被门禁阻止。large cache、query probability 和 checkpoint 按仓库规则不提交 Git。

## 9. 机制判断

失败不是因为 semantic head 完全学不到背景区分：后期 AUPRC 已明显提升；根因是少数 covered component 的候选 descriptor 与背景仍高度混淆。降低阈值可以救回这些 component，却会接收大量背景候选；提高阈值可以恢复 old hard gate 的低 Fa，却重新丢失约 4–5 个 component。仅靠同一候选 descriptor 上的 threshold、group rescue、utility 或 coverage loss，当前验证集上没有形成满足门禁的 Pareto 交点。

这支持后续研究转向“提高候选覆盖/少数困难组件的可分性”，而不是继续为同一 451-d descriptor 叠加更复杂的拒绝头。
