# MicroQuery-SAM 端到端完整训练实验记录

日期：2026-08-27

状态：**四组正式 100-epoch 核心训练、统一评测、反事实和 2,000 次 paired bootstrap 已完成；结论为 Useful Partial Success（单种子 validation，非论文主模型结论）**

## 1. 目的与比较边界

本实验检验：冻结 A1-P ImageEncoderViTHQ 与单视图 neck SpatialProbeHead 候选后，候选独立解码、可微 object gate 和 candidate sparse token 是否能在完整 100-epoch 联合训练中改善最终红外小目标分割，而不是只在冻结 query-mask cache 上改变后处理结果。

四个 matched runs 为：

| 变体 | Query 形式 | 最终 gate | candidate token 是否进入 decoder | 训练辅助项 |
|---|---|---|---|---|
| C0 | K=10 个点组成一个 query、一个 mask | 不适用 | 否 | full + covered |
| C1 | K 个独立 query、K 个 mask | valid 全 1 | 否 | 与 F1/F2 相同 |
| F1 | K 个独立 query、K 个 mask | predicted soft gate | 否 | 与 C1/F2 相同 |
| F2 | K 个独立 query、K 个 mask | predicted soft gate | 是 | 与 C1/F1 相同，另加低权重 token norm |

C1/F1/F2 具有完全相同的 MicroQuery head 架构、参数量和随机初始化。F1 与 F2 的唯一区别是 F2 把 `0.02 × LayerNorm(hidden)` 追加到原生 point sparse embedding 后；F1 也实例化同一 token 分支，但不消费它。

## 2. 数据与冻结证据

| 项目 | 固定值或 SHA-256 |
|---|---|
| 数据集 | IRSTD-1k |
| train / validation | 720 / 80 |
| test | **未读取** |
| seed | 20260825 |
| K | 10 |
| train split | `539be5c2c08eeddac03b8c59e57de30fe3d541e2c399163c3cade82124a7d9af` |
| validation split | `3f0206fda5f471690f47570f80990e81059e2282caa8739b04905879c29faa1e` |
| A1-P best-mask | `6320c5e2a68aa934b92b869998d826463b630f560f96e4257391deebabc9a904` |
| baseline EfficientSAM weights | `dff858b19600a46461cbb7de98f796b23a7a888d9f5e34c0b033f7d6eb9e4e6a` |
| train candidate cache | `9d36faa138915023cb265523483202317d81069096519e1d8c01c14504b72ffd` |
| validation candidate cache | `4269eb06c7f7d7dcec1edc55cdf4676bca4da700e95c58ad2963db5451766230` |
| shared head initialization | `547c3590d9a5d627a57cf0e054defd1b1bb612a79a55df9f943eb50a2235d6f5` |

候选 cache 经代码强制只能包含 `image_names/candidate_xy/candidate_scores/candidate_valid` 四项，且图像顺序必须与 split 完全一致。训练集有效候选中 target-like 1,040 个、background 1,846 个，另有 4,314 个 invalid slots。GT 只用于 forward 之后的 assignment、loss 和 metrics；`forward_deployable()` 不接受 GT、mask、semantic 或 component 参数。

## 3. 架构与训练协议

- ImageEncoderViTHQ：加载 A1-P 后全程 `eval + no_grad`；
- PromptEncoderHQ 与 MaskDecoderHQ：所有四组均可训练；
- MicroQuery head：451→256→256、object logits、candidate token，183,433 参数，小于 0.5M；
- optional adapter、multi-scale fusion、detail enhancer、AMGD、DoG-AMGD、HLDF、center decoder、task token 和 text embedding 全部关闭；
- ROI descriptor：shallow 7×7 与 neck 3×3 ROIAlign 后均值池化，再拼接归一化 xy 和 raw score，共 451 维；
- GT assignment：2 px 膨胀命中或 centroid distance≤3 px；多重命中取最近 centroid；primary/duplicate 均为 target-like；
- 增强：只用同步 horizontal/vertical flip，各 `p=0.5`；禁止 crop、随机 resize、rotation、gamma、noise、blur 和 candidate 修正；
- optimizer：AdamW，head `3e-4/1e-4`、prompt encoder `1e-5/1e-2`、mask decoder `2e-5/1e-2`；
- schedule：5-epoch warm-up 后 cosine decay，最小 LR 分别为 `3e-6/1e-6/1e-6`；
- batch：physical 4、gradient accumulation 1、effective 4；四组统一；
- BF16 autocast、loss FP32、gradient clipping 1.0、query chunk 5；
- 每 epoch 在固定 validation 80 张、threshold=0.5 评测；完整运行 100 epochs，不早停；
- 主 checkpoint：fixed-0.5 global IoU 最高；另保存 best mask AUPRC 和 last。

## 4. 实现与验证

新增统一核心模块、数据边界、训练、阈值评测、反事实、paired bootstrap 和 Windows/Linux launcher。四个变体共用同一训练脚本，没有复制四份训练逻辑。

单元测试覆盖计划列出的 20 个文件；增加 meaningful counterfactual effect 回归断言后，全仓库测试为 **94 passed**。真实单批次 F2 前反向确认：ImageEncoder 梯度为 0，PromptEncoder、MaskDecoder、object head、candidate token LayerNorm 与 learnable token scale 梯度均非零。四个 train720/val80 的 1-epoch smoke 均完成，无 NaN、Inf、OOM 或 shape 错误。

## 5. 100-epoch validation 结果

四组均完成 100 epochs，每轮评测 validation，且主 checkpoint 始终按固定 segmentation threshold=0.5 的 global IoU 选择。下表为 checkpoint 选择时的固定 0.5 结果；`Fa` 单位为 `×10^-6`，所有其他指标为百分数。

| Run | Best epoch | IoU | nIoU | F1 | Pd | Fa | mask AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 2 | 56.64 | 47.48 | 59.27 | 83.76 | 28.99 | 63.51 |
| C1 | 18 | **57.71** | **52.61** | **65.18** | **90.60** | 33.76 | 62.60 |
| F1 | 16 | 57.24 | 52.43 | 63.93 | 86.32 | **22.70** | **66.28** |
| F2 | 1 | 56.68 | 47.73 | 60.15 | 85.47 | 37.38 | 64.48 |

训练后仅在同一 validation 上扫描 `0.05:0.05:0.95`。每组有 19 个阈值点；选择阈值后的正式主表如下。

| Run | Best epoch | 阈值 | IoU | nIoU | F1 | Pd | Fa | mask AUPRC | CTR | 延迟 ms/image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 2 | 0.75 | 57.07 | 47.71 | 59.28 | 82.05 | **27.08** | 63.51 | 84.82 | **22.29** |
| C1 | 18 | 0.40 | 57.75 | **53.28** | **65.60** | **90.60** | 34.71 | 62.60 | **94.64** | 40.02 |
| F1 | 16 | 0.15 | **58.62** | 51.72 | 64.13 | **90.60** | 32.42 | **66.28** | **94.64** | 39.91 |
| F2 | 1 | 0.40 | 56.74 | 48.49 | 61.18 | 86.32 | 39.10 | 64.48 | 89.29 | 40.07 |

关键解释：

1. C1 相对 C0 的 IoU/nIoU/F1/Pd 分别提高 0.67/5.56/6.32/8.55pp，但 Fa 增加 7.63，说明 independent queries 与逐 query 监督显著提高了 covered-target recovery，同时增加计算和虚警；MicroQuery 的后续归因必须以 C1 而不是 C0 为 matched control。
2. F1 相对 C1 的 global IoU 提高 0.87pp、Fa 降低 2.29（6.59%）、Pd 持平，但 nIoU/F1 分别下降 1.56/1.47pp。这是混合胜负，而不是全面改善。
3. F2 相对 F1 的 IoU/nIoU/F1/Pd 分别下降 1.88/3.23/2.95/4.27pp，Fa 增加 6.68；其 best checkpoint 退回 epoch 1，candidate token 没有形成稳定训练收益。

### 5.1 Paired bootstrap

逐图配对 bootstrap 为 2,000 次、95% CI。下表的 IoU/nIoU/Pd/CTR/逐图 mean AUPRC 均为百分点，Fa 仍为 `×10^-6`；这里的 AUPRC 是逐图 AUPRC 的配对均值，不与主表的全 validation 像素级 AUPRC 混用。

| Comparison | ΔIoU [95% CI] | ΔnIoU [95% CI] | ΔPd [95% CI] | ΔFa [95% CI] | ΔCTR [95% CI] | Δ逐图 mean AUPRC [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
| C1−C0 | +0.67 [−1.43,+3.05] | +5.56 [+0.95,+10.77] | +8.55 [+3.13,+15.20] | +7.63 [−4.58,+20.60] | +9.82 [+4.39,+16.19] | +2.43 [−1.50,+6.70] |
| F1−C1 | +0.87 [−2.02,+5.43] | −1.56 [−3.28,+0.12] | 0.00 [−2.59,+2.48] | −2.29 [−4.96,−0.19] | 0.00 [−2.66,+2.54] | +0.35 [−1.32,+1.84] |
| F2−F1 | −1.88 [−8.04,+2.67] | −3.23 [−6.10,−0.66] | −4.27 [−8.77,0.00] | +6.68 [−2.10,+17.17] | −5.36 [−9.48,−1.80] | −1.80 [−5.90,+1.60] |
| F1−C0 | +1.55 [−2.41,+7.27] | +4.00 [−0.16,+8.68] | +8.55 [+3.17,+14.75] | +5.34 [−6.30,+17.74] | +9.82 [+4.42,+16.04] | +2.78 [−0.41,+6.92] |

F1−C1 的 global-IoU CI 跨 0，nIoU 方向为负且接近显著；逐图 F1 差值为 −1.47pp，95% CI `[−2.96,−0.13]`，明确不支持“F1 全面优于 C1”。Fa 的下降 CI 不跨 0，但绝对值较小。

## 6. 反事实、效率与统计

计划中的“明显优于”原本没有数值操作定义。结果审计发现仅用浮点数 `correct > shuffled` 会把 F2 的 1–3 像素差异误判为 token 有效。最终采用保守、可审计规则：在同一 validation-selected threshold 下，对每个必需 intervention，correct 的 global IoU、mean nIoU 和 mask AUPRC 必须都更高，且至少一项提高 `0.005`（0.5pp），才算 meaningful pass；同时另存纯 ordering 结果。该规则是对原计划模糊条款的事后保守操作化，不伪装成预注册数值门槛。

### 6.1 F1 gate counterfactual

| Gate 条件 | IoU | nIoU | F1 | Pd | Fa | mask AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| Correct | **58.62** | **51.72** | 64.13 | 90.60 | **32.42** | **66.28** |
| All-one | 58.55 | 51.60 | **64.19** | **91.45** | 36.24 | 62.83 |
| Batch-shuffled | 56.45 | 49.80 | 62.34 | 82.91 | 31.85 | 58.23 |
| Candidate-shuffled | 37.57 | 24.06 | 31.90 | 50.43 | 23.46 | 40.15 |
| Inverted | 16.29 | 6.20 | 9.00 | 19.66 | 15.26 | 32.26 |

F1 gate 的 ordering 与 meaningful-effect 均通过。Correct 相对 all-one 的 global IoU 只高 0.07pp，但 mask AUPRC 高 3.44pp，Fa 低 10.53%，且 shuffled/inverted 显著破坏性能，说明 gate 内容确实被最终 mask 消费。不过 all-one 的 Pd 高 0.85pp、F1 高 0.07pp，仍体现拒绝安全性的边界。

### 6.2 F2 token counterfactual

| Token 条件 | IoU | nIoU | F1 | Pd | Fa | mask AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| Correct | 56.740 | 48.486 | 61.181 | 86.325 | 39.101 | 64.479 |
| Zero | 56.702 | 48.484 | 61.179 | 86.325 | 39.101 | 64.473 |
| Batch-shuffled | 56.724 | 48.486 | 61.181 | 86.325 | 39.101 | 64.473 |
| Candidate-shuffled | 56.719 | 48.472 | 61.161 | 86.325 | 39.291 | 64.472 |
| Random | 56.664 | 48.446 | 61.145 | 86.325 | 39.291 | 64.469 |
| Coordinate-only | 56.504 | 48.304 | 61.002 | 86.325 | 40.245 | 64.488 |

F2 的纯 ordering 为真，但 meaningful-effect **未通过**：correct 相对 zero、batch-shuffled、candidate-shuffled 的最大三指标增益仅 0.038/0.016/0.022pp，远小于 0.5pp。token 分支虽有非零梯度，却没有提供可测的候选语义作用，F2 的差异只能视为附加容量/优化噪声。F1/F2 的 coordinate counterfactual 均通过；随机背景坐标仅由离线诊断使用 GT 构造，deployable forward 只接收构造后的坐标。

### 6.3 效率

| Run | Encoder ms | Cached probe ms | ROI head ms | Prompt+decoder ms | Decoder calls/image | 总延迟 ms/image | Peak GPU MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 14.69 | 0 | 0 | 7.60 | 1 | **22.29** | 591.60 |
| C1 | 14.69 | 0 | 0.91 | 24.37 | 10 | 40.02 | 592.65 |
| F1 | 14.73 | 0 | 0.67 | 24.06 | 10 | 39.91 | 592.65 |
| F2 | 16.26 | 0 | 0.66 | 24.13 | 10 | 40.07 | 592.68 |

候选坐标来自冻结 cache，因此正式评测的 probe latency 为 0，并明确标注不是在线 probe 调用。相对 C0，独立 K=10 query 将延迟提高约 79%；F1 相对 C1 没有额外可测延迟，F2 也没有显著额外开销。

### 6.4 分级决策

| 决策项 | 结论 |
|---|---|
| Strong Success | 否；F1 nIoU 下降 1.56pp，且相对 C0 的 Fa 更高 |
| Useful Partial Success | **是，类型 A**；F1 相对 C1 global IoU +0.87pp、Fa −6.59%、Pd 持平，gate 反事实通过 |
| Winning variant | F1 soft gate |
| Gate 被消费 | 是，但收益混合且 global-IoU CI 跨 0 |
| Token 被消费 | 有梯度但无 meaningful counterfactual effect，否 |
| 进入三随机种子 | 否；计划只在 Strong Success 时进入三种子 |
| NUAA | 本轮未执行；Partial 仅允许后续增加一个 IRSTD seed 与一个 NUAA 单种子确认 |
| Test consumed | **否** |

严格按自动分级规则，F1 属于 Useful Partial Success；统计上只能支持“soft gate 改变了低虚警—分割权衡，且 gate 内容被消费”，不能支持“稳定全面优于 matched independent-query control”。F2 candidate token 被拒绝为有效组件。核心必跑阶段到此结束，不扩展 test、三种子或 NUDT-SIRST。

## 7. 证据边界

本阶段所有模型选择、阈值选择、反事实和统计均限于 validation，test 未读取。四个正式 manifest、resolved config、train history、三个 checkpoint、19 点阈值/Pd-Fa 曲线、逐图/逐 query/逐组件/面积分层、反事实和效率产物位于 `outputs/microquery/end2end_full/IRSTD-1k/`；大 checkpoint 与 cache 受 `.gitignore` 管理，不提交 Git。

四个主运行每个都有 80 条逐图、117 条逐组件和 344 条有效候选记录；全程无 NaN、Inf、OOM、shape error 或错误日志。C1/F1/F2 的共享 head 初始化 SHA 均为 `547c...d6f5`；F2 的 candidate-token LayerNorm/token-scale 梯度和分别为 `0.001873/0.003296`，而 C1/F1 未消费 token 时二者均为 0。
