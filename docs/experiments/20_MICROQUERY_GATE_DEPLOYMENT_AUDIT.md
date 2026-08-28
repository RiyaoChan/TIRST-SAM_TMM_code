# MicroQuery Gate 部署审计与跨数据集启动决策

日期：2026-08-28

状态：**14 个零训练 gate 条件、19 点阈值扫描、matched-Pd/Fa、Pareto、面积分桶、2,000 次 paired bootstrap、反事实、失败案例和 online probe replay 均已完成；按预注册安全规则判定为 Gate Failure。**

## 1. 最终结论

本阶段最重要的结论不是“某个 gate 的 Fa 最低”，而是：

> C1/F1 的 object gate 确实被最终 mask 消费，也能降低 Fa；但没有一个显式固定 gate 同时满足 Pd、tiny 1–9 px 和 CTR 的预注册安全限制。因此停止 soft-gate 主线，保留 C1 independent-query + per-query supervision，不启动 gate 的 IRSTD 第二 seed 或 NUAA 训练。

`R4 = residual(ρ=0.2, T=1.5)` 是所有显式条件中平均 matched-Pd Fa 最低的**诊断 fallback**，不是可发布的 winning gate：

- F1：Fa 从 33.38 降至 27.66 ×10^-6（−17.14%），但 Pd 从 91.45% 降至 89.74%，tiny-Pd 从 84.75% 降至 81.36%，即少检 2 个 1–9 px component；CTR 从 95.54% 降至 93.75%，下降 1.79pp；
- C1：nIoU/F1 从 51.70%/64.40% 提高到 53.76%/66.10%，tiny-Pd 也提高 1.69pp，但 Fa 从 27.47 增至 28.80 ×10^-6，没有背景抑制收益；
- 所有显式条件均至少违反一项安全约束，故 `paper_safe=false`。

## 2. 冻结边界与证据身份

本阶段没有训练、没有修改 checkpoint、没有修改 head/encoder/decoder、没有改变 K/NMS/候选坐标，也没有读取 test。

| 项目 | 固定值或 SHA-256 |
|---|---|
| Base commit | `22b4e1a21573abc60191f38bb8a93305e0733d3e` |
| 数据集 / split | IRSTD-1k validation，80 张，117 components |
| seed / K | `20260825` / `10` |
| C1 checkpoint | epoch 18，`8da90be39c0067b031b801318452e3466aa08bb0b5997fa2cffc456677218791` |
| F1 checkpoint | epoch 16，`63f90d0d849e7bf27408413600c787ab20d93796fc259750e2f3bb05693b4632` |
| validation split | `3f0206fda5f471690f47570f80990e81059e2282caa8739b04905879c29faa1e` |
| validation candidate cache | `4269eb06c7f7d7dcec1edc55cdf4676bca4da700e95c58ad2963db5451766230` |
| A1-P checkpoint | `6320c5e2a68aa934b92b869998d826463b630f560f96e4257391deebabc9a904` |
| neck probe checkpoint | `cc96c90cd19f4b215c535656055e5788b5d9db5cbe2f1fbc2910bb87faa0ae47` |
| baseline EfficientSAM weights | `dff858b19600a46461cbb7de98f796b23a7a888d9f5e34c0b033f7d6eb9e4e6a` |
| Test consumed | **否** |

GT 只在 forward 之后用于 assignment、面积分桶和指标。`forward_deployable()` 仍不接受 GT、mask、target、component、semantic 或 supervision 参数。

## 3. Gate API 修复

旧实现用 checkpoint epoch 同时隐式决定训练 warm-up 和推理 gate，且 C1 在 forward 中由 variant 强制 all-one。现已拆分为：

- 训练态：必须显式传入 `training_epoch`，继续使用原训练 warm-up；
- 评测/部署态：C1/F1/F2 必须显式传入不可变的 `GateDeploymentConfig`；缺失配置直接抛错；
- 部署模式：`all_one`、`raw`、`residual`、`legacy_checkpoint_schedule`；
- `legacy_checkpoint_schedule` 只用于复现旧结果，不能作为跨数据集部署配置；
- C1 checkpoint 可直接使用 predicted gate，不伪装成 F1，也不改 checkpoint；
- C1/F1 在同一显式配置下执行完全相同的 gate 公式。

部署重放进一步支持把一次 image encoder 结果同时交给 online probe 与 MicroQuery，避免为了在线候选重复调用 encoder。

## 4. 精度协议与旧结果复现

online candidate cache 原本由 float32 image encoder 生成。为保证真正的“单次 encoder → probe → MicroQuery → SAM”链路，审计最终固定为：

```text
Image encoder: float32，一次
Probe: 复用同一 encoder features
MicroQuery head / PromptEncoderHQ / MaskDecoderHQ: bfloat16 autocast
```

若把 image encoder 也放入 bfloat16 autocast，候选坐标 cache 仍相同，但阈值附近会出现 1 个 component 的离线/在线差异。该路径已被拒绝。统一协议后离线与在线的最终指标逐项完全一致。

旧正式记录采用整条 forward 的 bfloat16 autocast，因此本阶段的数值有轻微浮动，但机制与结果在允许范围内复现：

| 回归点 | 旧记录 | 本阶段 deployable protocol |
|---|---:|---:|
| C1-A0 @0.40 IoU / nIoU / Pd / Fa | 57.75 / 53.28 / 90.60 / 34.71 | 57.65 / 53.18 / 90.60 / 34.71 |
| F1-L @0.15 IoU / nIoU / Pd / Fa | 58.62 / 51.72 / 90.60 / 32.42 | 58.60 / 51.76 / 90.60 / 32.81 |

F1 的 epoch-16 legacy 参数为 `ρ=0.2, T=1.4827586`；C1 的 epoch-18 LegacyPredicted 为 `ρ=0.1, T=1.4137931`。当前正式 C1 基线始终是 all-one，C1-L 只用于对称零训练诊断。

## 5. 14 条件完整矩阵

每个条件均报告 checkpoint anchor threshold、固定 0.5、`0.05:0.05:0.95` 19 点曲线、matched-Pd/Fa 与 Pareto。下表 `MP` 为 `Pd≥105/117=89.7436%` 时的最小 Fa 点；Fa 单位为 `×10^-6`，其他性能指标为百分数。

|ID|ρ|T|Anchor IoU|Anchor nIoU|Anchor F1|Anchor Pd|Anchor Fa|MP th|MP IoU|MP nIoU|MP F1|MP Pd|MP Fa|tiny Pd|CTR|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|C1-A0|1.00|1.0000|57.65|53.18|65.53|90.60|34.71|0.95|56.74|51.70|64.40|89.74|27.47|81.36|93.75|
|C1-R0|0.00|1.0000|54.56|53.11|65.05|85.47|37.38|-|-|-|-|-|-|-|-|
|C1-R1|0.10|1.0000|54.59|53.19|65.13|85.47|39.67|0.10|57.51|52.95|65.37|90.60|34.52|83.05|94.64|
|C1-R2|0.20|1.0000|54.76|53.59|65.48|87.18|40.05|0.20|57.61|53.57|65.89|90.60|32.42|83.05|94.64|
|C1-R3|0.10|1.5000|54.75|53.64|65.50|86.32|40.05|0.15|57.70|54.01|66.31|90.60|30.71|83.05|94.64|
|C1-R4|0.20|1.5000|54.93|53.20|65.14|88.03|25.75|0.25|57.26|53.76|66.10|89.74|28.80|83.05|93.75|
|C1-L|0.10|1.4138|54.75|53.64|65.50|86.32|40.05|0.15|57.25|53.74|66.06|89.74|30.52|83.05|93.75|
|F1-A0|1.00|1.0000|58.51|51.61|64.19|91.45|36.24|0.30|58.09|51.86|64.34|91.45|33.38|84.75|95.54|
|F1-R0|0.00|1.0000|57.66|51.43|63.15|87.18|27.08|0.05|58.45|51.73|63.88|89.74|28.80|83.05|93.75|
|F1-R1|0.10|1.0000|58.34|51.63|63.56|88.03|28.04|0.10|58.65|51.87|64.31|90.60|31.47|83.05|94.64|
|F1-R2|0.20|1.0000|58.48|51.54|63.78|89.74|32.42|0.20|58.14|51.72|63.89|89.74|30.14|81.36|93.75|
|F1-R3|0.10|1.5000|58.72|52.05|64.18|89.74|29.18|0.20|58.32|52.24|64.32|89.74|27.28|81.36|93.75|
|F1-R4|0.20|1.5000|58.60|51.76|64.15|90.60|32.81|0.25|58.20|52.29|64.38|89.74|27.66|81.36|93.75|
|F1-L|0.20|1.4828|58.60|51.76|64.15|90.60|32.81|0.25|58.20|52.29|64.38|89.74|27.66|81.36|93.75|

R0 在 C1 上没有任何阈值达到 Pd floor。R1–R4 虽能在两模型上找到 matched-Pd 点，但没有通过完整安全筛选。

## 6. 预注册安全选择与 2×2 归因

安全条件为：Pd 至多损失 1 个 component、tiny 1–9 px 至多损失 1 个 component、CTR 相对各自 all-one 下降不超过 0.85pp、nIoU/F1 下降不超过 0.5pp。

| 显式条件 | 安全结果 | 主要失败原因 |
|---|---|---|
| R0 | 失败 | C1 无 matched-Pd；F1 CTR −1.79pp |
| R1 | 失败 | F1 CTR −0.89pp，略超过 0.85pp 门槛 |
| R2 | 失败 | F1 tiny 少 2 个；CTR −1.79pp |
| R3 | 失败 | F1 tiny 少 2 个；CTR −1.79pp |
| R4 | 失败 | F1 tiny 少 2 个；CTR −1.79pp |

R4 的 2×2 归因说明 gate 训练并没有产生稳定全面优势：

| Training | All-one inference（matched-Pd） | R4 predicted gate（matched-Pd） |
|---|---|---|
| C1：训练 final mask 未用 gate | IoU 56.74 / nIoU 51.70 / F1 64.40 / Pd 89.74 / Fa 27.47 | IoU 57.26 / nIoU 53.76 / F1 66.10 / Pd 89.74 / Fa 28.80 |
| F1：训练 final mask 使用 gate | IoU 58.09 / nIoU 51.86 / F1 64.34 / Pd 91.45 / Fa 33.38 | IoU 58.20 / nIoU 52.29 / F1 64.38 / Pd 89.74 / Fa 27.66 |

同一 R4 下，F1 相对 C1 的 IoU/Fa 更好，但 nIoU/F1 更差；F1−C1 的 F1 bootstrap 差值为 −1.72pp，95% CI `[−3.82,−0.01]pp`。这不支持“F1 gate training 稳定优于 C1 decoder”的结论。

## 7. Tiny-target、安全性与统计

matched-Pd 面积分桶的关键差异为：

| Model | Gate | 1–9 px Pd | 10–16 px Pd | 17–25 px Pd | >25 px Pd |
|---|---|---:|---:|---:|---:|
| C1 | A0 | 81.36 | 100.00 | 100.00 | 93.33 |
| C1 | R4 | **83.05** | 100.00 | 100.00 | 86.67 |
| F1 | A0 | **84.75** | 100.00 | 100.00 | 93.33 |
| F1 | R4 | 81.36 | 100.00 | 100.00 | 93.33 |

C1-R4 增加 1 个 tiny 检出，但丢失 1 个 >25 px component；F1-R4 相对 F1-A0 丢失 2 个 tiny components。gate 的主要风险确实集中在弱小目标，而不是候选覆盖不足：Coverage@10 为 95.73%，tiny Coverage@10 为 91.53%，所有条件共享同一坐标。

2,000 次 image-level paired bootstrap：

- C1-R4−C1-A0：ΔnIoU `+2.05pp [ +0.25,+4.12 ]`，ΔF1 `+1.69pp [ +0.21,+3.37 ]`；ΔFa `+1.33 [−2.67,+5.53] ×10^-6`，没有 Fa 改善；
- F1-R4−F1-A0：ΔFa `−5.69 [−13.54,−0.76] ×10^-6`，但 ΔPd `−1.68pp [−4.46,0.00]`，Δtiny-Pd `−3.33pp [−8.77,0.00]`；
- F1-R4−C1-R4：ΔIoU CI 跨 0，ΔnIoU 为负且大部分质量在 0 以下，ΔF1 `−1.72pp [−3.82,−0.01]`。

所以 F1 的 Fa 改善是真实方向，但伴随预注册禁止的 tiny/Pd 损失；不能只摘取 Fa 写正结果。

## 8. Gate 反事实

R4 在其 matched-Pd threshold=0.25 下的反事实如下：

| Model | Gate 条件 | IoU | nIoU | F1 | Pd | Fa ×10^-6 |
|---|---|---:|---:|---:|---:|---:|
| C1 | correct | 57.26 | 53.76 | 66.10 | 89.74 | 28.80 |
| C1 | candidate-shuffled | 35.04 | 18.16 | 24.84 | 41.03 | 20.22 |
| C1 | batch-shuffled | 48.66 | 45.89 | 57.65 | 72.65 | 26.13 |
| C1 | inverted | 21.29 | 7.66 | 10.32 | 18.80 | 29.37 |
| C1 | all-one | 57.56 | 53.18 | 65.47 | 90.60 | 36.43 |
| F1 | correct | 58.20 | 52.29 | 64.38 | 89.74 | 27.66 |
| F1 | candidate-shuffled | 33.92 | 17.55 | 24.28 | 42.74 | 19.65 |
| F1 | batch-shuffled | 48.62 | 45.50 | 57.65 | 76.07 | 48.45 |
| F1 | inverted | 13.16 | 4.68 | 6.65 | 15.38 | 13.16 |
| F1 | all-one | 58.09 | 51.74 | 64.25 | 91.45 | 34.14 |

correct 与 shuffled/inverted 明显不同，证明 gate 内容被 decoder 输出的 gated aggregation 消费。这个因果结果只能说明“gate 有作用”，不能推翻 safety failure。

## 9. Online probe replay 与效率

同一 80 张 validation、相同 probe/NMS/score/K 的 replay 结果：

| 检查项 | 结果 |
|---|---:|
| valid exact | 100% |
| coordinate within 0.5 px / 1 px | 100% / 100% |
| rank agreement within 1 px | 100% |
| coordinate mean / max distance | 0 / 0 px |
| score mean / max absolute error | 0 / 0 |
| Coverage@10 | 95.73%（112/117） |
| C1/F1 使用相同 online candidates | 是 |
| online final metrics 与离线审计 | **逐项完全相同** |

RTX 5090 D、batch=1、warm-up 10、重复 50 次：

| 组件 | ms/image |
|---|---:|
| float32 image encoder | 15.09 |
| online neck probe | 2.28 |
| MicroQuery head + 10-query Prompt/Mask decoder | 26.76 |
| 组件和 / 实测 end-to-end | 44.14 / 44.47 |

实测总链路只有 1 次 encoder 调用，peak GPU memory 为 575,387,136 bytes。

## 10. 最终分级与下一步

| 决策项 | 结论 |
|---|---|
| Strong Gate Success | 否；没有 paper-safe 显式配置 |
| Useful Gate Calibration | 否；R4 的 Fa 收益伴随 2 个 tiny component 和 1.79pp CTR 损失 |
| Gate Failure | **是** |
| C1 + predicted gate 是否保留 | 否；C1-R4 的 Fa 高于 C1-A0 matched-Pd |
| F1 gate training 是否具有独立稳定价值 | 否；质量指标混合，tiny safety 失败 |
| IRSTD 第二 seed / NUAA gate 训练 | **不启动** |
| F2 | 永久停止 |
| 保留主线 | C1 independent-query + per-query foreground/background supervision，部署 all-one |
| Test consumed | **否** |

这次审计推翻了“只看 F1 Fa 就继续 gate”的冲动：gate 能工作，但不能安全工作。下一步若继续 MicroQuery，应研究不会压掉 candidate identity/tiny target 的显式 no-object 或独立 query 监督，而不是继续调 `ρ/T`。

## 11. 产物与复现

本地产物位于：

```text
outputs/microquery/gate_deployment_audit/IRSTD-1k/
```

包含 14 个条件各自的 resolved config、fixed/anchor/matched summary、threshold/Pareto 曲线、逐图/逐组件/逐 query、gate distribution、面积分桶；另有 5 组 bootstrap、2×2 归因、反事实、7 类失败案例、online replay 与 latency。大体积输出受 `.gitignore` 管理，Git 中提交代码、测试与本中文记录。

复现：

```powershell
$env:PYTHONPATH='.'
python scripts/audit_microquery_gate_deployment.py --device cuda --bootstrap_repeats 2000
python scripts/eval_microquery_online_probe.py --device cuda --warmup 10 --repeats 50
pytest -q
```

全仓库测试：**115 passed**。
