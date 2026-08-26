# Top-3 Self-Prompt 最小实验计划

状态：**仅设计，未运行，不包含性能结果。** 当前阶段不修改主模型、不启动长训练。每个方案先用缓存和oracle反事实证伪，只有过闸门才写训练代码。

Top-3：

1. **MicroQuery-SAM**：一候选一query、独立micro-mask与no-object；
2. **TB-Prompt**：候选级target–background成对prompt；
3. **RQ-Adapt**：由首轮SAM response驱动refine/reject。

## 1. 共同控制与协议

### 数据与复现

- 主筛选：IRSTD-1k现有独立train/validation/test；validation只选结构/阈值，test只在最终候选通过后一次性评估。
- 第二筛选数据集：NUAA-SIRST或NUDT-SIRST必须沿官方/现有clean split；不得用test搜索prompt阈值。
- 推理严格无GT point/box/mask；oracle只作为单独上界行，不能与可部署结果混表。
- deterministic resize；seed固定并记录，筛选至少3 seeds后才能称稳定。
- backbone、image encoder checkpoint、输入分辨率、segmentation threshold搜索范围一致。
- 同参数量/同decoder-call控制：多query的共享image embedding不能重复算encoder；额外adapter容量给baseline一个同参数无prompt MLP/FPN控制。
- 强机制控制：SAM-SPL式shallow dense visual prompt、DVPT式global visual token、AutoPromptSeg式低不确定性外部排序分别作为“视觉prompt/UQ筛点”对照；不能把重型MUP-SAM fusion的收益混入prompt贡献。
- 面积分桶：1–9、10–16、17–25、>25 pixels。

### 必报 mask 指标

`Pd、Fa(×10^-6)、global IoU、nIoU、F1、mask AUPRC`，同时报告latency、峰值显存、trainable/total parameters和decoder calls/image。

### 必报 prompt/query 指标

- Component Recall@K；Prompt Precision@K；False Prompts/MP；
- duplicate prompts/component；candidate AUPRC；
- no-object accuracy、AUROC/AUPRC；ECE、Brier；
- risk–coverage；prompt-to-mask sensitivity；
- oracle center/box/micro-mask spatial prompt upper bound；
- 多目标图的component coverage和query collision rate。

### 所有方案的反事实矩阵

| 条件 | 定义 | 目的 |
|---|---|---|
| correct | 预测candidate/query按原图配对 | 正常部署 |
| shuffled | batch内打乱query或background，坐标/容量不变 | 测模型是否消费内容 |
| zero | query/token/micro-mask置零 | 测是否退化为空prompt |
| wrong/background | 给真候选错误背景或给背景候选target token | 测错误prompt安全性 |
| oracle | GT component center/box/micro-mask，仅上界 | 定位generator还是decoder瓶颈 |
| candidate drop | 逐个移除query | 测边际credit/harmful prompt |
| reliability zero/random | 保持坐标，改权重 | 测可靠性是否进入decoder |
| one vs multi query | `[B,1,K,2]` vs `[B,K,1,2]` | 测候选污染 |
| shared/specific/shuffled role | 共享point type、target/background专用type、随机交换type | 按S4M证据检验角色身份是否被消费 |

### 通用停止线

- oracle prompt改变不了mask：停止prompt generator，先做response/domain adaptation；
- shuffled/zero与correct的mask delta近零：当前接口没有消费prompt，停止蒸馏/复杂query；
- prompt-level改善不能在两个数据集映射到任一mask指标且Fa/Pd权衡不优：停止；
- 只有单seed/test-set调参有效：作废；
- 新方案推理需要GT/reference/text而未单列：作废。

## 2. Experiment M：MicroQuery-SAM

### M0：零训练 shape 反事实（最高优先级）

复用A3 best-mask checkpoint与同一candidate cache，只改decoder batch组织：

```text
M0-a  one-query:   points [B,1,K,2] → mask [B,1,H,W]
M0-b  multi-query: points [B,K,1,2] → masks [B,K,H,W]
```

聚合四种：

1. unweighted mask max；
2. candidate-score weighted max/sum；
3. reliability weighted；
4. reliability × SAM predicted-IoU weighted。

同时用oracle center points重复M0，区分“候选错”与“query混合错”。

**通过线**：至少一种multi-query聚合在两个validation split中相对one-query满足：Fa下降≥10%且Pd下降≤0.5pp，或global IoU/nIoU提升≥0.5pp且Fa不恶化>5%。oracle multi-query必须比oracle one-query有同方向证据。

**停止线**：oracle multi-query都无收益；或multi-query只增加重复mask/显存，prompt-to-mask sensitivity不提高。

### M1：Oracle prompt representation ladder

对相同GT components比较：

| Run | 每candidate prompt | 目的 |
|---|---|---|
| M1-P | center positive point | 坐标上界 |
| M1-PN | center + GT外环背景点 | 背景点上界 |
| M1-B | tight box | 尺度/边界上界 |
| M1-M | downsampled local GT micro-mask | shape上界 |
| M1-Q | learned/free query initialized at center | latent query接口 |

若M1-M不优于M1-P/M1-B，不实现micro-mask head；若所有上界都接近null prompt，直接转Experiment R的response adapter诊断。

### M2：最小trainable set head（仅M0/M1通过后）

固定image encoder和现有candidate generator。对每candidate ROIAlign shallow feature，单层MLP输出`query[256]`、`object_logit`；可选2层轻量micro-mask head。K≤8，query维度与EfficientSAM一致。

消融：

| Run | Independent query | `∅` | micro-mask | reliability进入聚合 |
|---|---:|---:|---:|---:|
| M2-0 | N | N | N | N（当前结构） |
| M2-1 | Y | N | N | N |
| M2-2 | Y | Y | N | N |
| M2-3 | Y | Y | Y | N |
| M2-4 | Y | Y | Y | Y |

loss：Hungarian component assignment、object/no-object CE（empty query权重单独调）、local/final Dice+BCE、duplicate overlap penalty；M2-4加Brier/ECE友好的quality loss。

**成功线**：M2-2必须证明`∅`本身有效（no-object accuracy、Fa）且M2-3在1–9/10–16桶至少一个recall/Pd指标提升；M2-4的reliability置零/随机应显著改变mask，否则它未被消费。

### 资源预算

- M0/M1：不训练，1个checkpoint、2个validation split，预计小时级；
- M2 sanity：固定encoder，20 epochs确认loss/shape；
- 只有过闸门才100 epochs×3 seeds；**当前不计划1000 epochs**，1000 epochs不能弥补机制闸门失败。

## 3. Experiment T：TB-Prompt

### T0：物理负点上界与安全性

不训练，针对每candidate独立query构造：

| Run | Target | Background | 反事实 |
|---|---|---|---|
| T0-a | center point | none | target-only |
| T0-b | center | 4点outer ring | 最简单paired spatial prompt |
| T0-c | center | shuffled candidate ring | 错背景 |
| T0-d | background candidate center | 真target ring | label inversion |
| T0-e | oracle target center | oracle clean background ring | 上界 |

ring半径按candidate局部尺度固定，不按test GT调；oracle行可用GT尺度但单独报告。

**通过线**：T0-b/e相对T0-a在Fa或False Prompts/MP下降≥10%，tiny Pd下降≤0.5pp；T0-c必须显著差于T0-b，否则background内容没有被消费。

**停止线**：负点普遍删除tiny target，或correct/shuffled ring无差异。

### T1：冻结 prompt encoder 的 paired latent condition

如果EfficientSAM prompt encoder允许mask/point embedding，比较：

1. `FeatureConcat`：target/background特征普通拼接后加到image feature；
2. `PointPN`：正/负物理点；
3. `PromptSpacePair`：inner/outer连续mask或latent states分别经过冻结prompt encoder；
4. `PromptSpacePair+AsymGate`：background state只作非对称抑制；
5. `ShuffledBG/WrongBG/ZeroBG`。
6. `SharedRole/SpecificRole/ShuffledRole`：target/background共用type、使用专用type、只打乱type而不改位置。

这一组直接对应IP-SAM与S4M的关键边界。若`PromptSpacePair`不优于同参数`FeatureConcat`，或`SpecificRole≈ShuffledRole`，不能主张prompt-space/角色机制。

### T2：candidate-level paired query（仅T0/T1通过后）

`q+`读取center+inner，`q-`读取outer ring+global background；二者不先相加。decoder前计算`gate = sigmoid(f(q-))`，作用于`q+`或dense micro-mask；object head输出target/background/no-object三态。

loss与压力测试：

- target/background margin；
- candidate mask/object loss；
- shuffled/wrong background应退化但不比target-only更危险；
- hard negatives：云边、建筑热源、强边缘、传感器坏点；
- 按1–9像素桶检查over-suppression。

**成功线**：T2相对同参数target-only在两个数据集Fa均降≥10%，Pd绝对下降≤0.5pp，且correct background显著优于shuffled/wrong；否则停止。

### 资源预算

- T0：零训练；
- T1：可先只训练≤0.5M adapter，冻结encoder/decoder，20 epochs；
- T2：过闸门后100 epochs×3 seeds；不与M2同时做长训练，先选机制更强者。

## 4. Experiment R：RQ-Adapt

### R0：Response 是否含可用拒绝信号

对A3 candidates逐个独立调用同一冻结decoder，缓存：

- SAM predicted-IoU；
- mask area、peak、entropy、compactness；
- candidate center与mask centroid距离；
- inner/outer mask contrast；
- mask feature与candidate query的attention/response token（若接口可得）；
- candidate score与view reliability作为基线。

用GT component只在analysis阶段给每candidate标TP/FP；训练/测试严格按split。比较单变量、logistic regression和同参数MLP的candidate AUPRC/ECE，并加入AutoPromptSeg式低不确定性PSS/NMS/Top-K与原candidate score作为外部排序强基线。

**通过线**：response-only或response+candidate相对candidate-score和UQ-ranking两种强baseline的AUPRC均提升≥3pp，且validation选择的reject threshold在第二数据集使Fa降≥10%、Pd下降≤0.5pp。

**停止线**：SAM-IoU/response与TP/FP无关，或oracle response reject也不改善最终mask。

### R1：Candidate drop / harmful prompt truth table

对K≤5样本计算：

```text
credit_k = Q(mask_all) - Q(mask_without_k)
```

`Q`只在训练/analysis用GT定义，可分别采用IoU、Pd–λFa或component F1。该GT-LOO只作为PromptPilot式oracle诊断，不能成为部署输入或“首次credit”主张；另记录无需GT的proxy（SAM-IoU、mask overlap、center agreement），检验其能否预测`credit_k<0`。

必须含：TP点、FP点、duplicate点、同图多target点、全背景图。若leave-one-out credit严重非加性，报告pair-drop而不是强行拟合scalar。

### R2：单次 Refine–Reject adapter（仅R0/R1通过后）

本阶段明确禁止复刻PromptPilot：不使用标注reference、不做多agent/RL、不在推理访问GT，也不把GT-LOO credit当输入。部署只消费当前图像、candidate和首轮冻结SAM response。

第一轮低分辨mask response编码为`r_k`，更新：

```text
q'_k = q_k + gate_k * Adapter([q_k, r_k, reliability_k])
action_k ∈ {accept, refine, reject}
```

只对`refine`候选做第二轮decoder；accept复用首轮mask，reject不聚合。设置cost penalty避免所有候选都refine。

消融：

| Run | response | reliability | action | second pass |
|---|---:|---:|---|---:|
| R2-0 | N | candidate score | accept/reject | N |
| R2-1 | Y | N | accept/reject | N |
| R2-2 | Y | Y | accept/reject | N |
| R2-3 | Y | Y | accept/refine/reject | selective |
| R2-shuf | shuffled response | Y | 同R2-3 | selective |

**成功线**：R2-2必须优于R2-0及UQ-ranking控制，R2-shuf必须退化；R2-3在Fa/Pd不劣时，相对R2-2的额外延迟需有显著mask收益。若R2-1≈R2-shuf或不优于candidate/UQ ranking，response未提供独立信号，立即停止。

### 资源预算

- R0/R1：仅缓存推理，小时级到1天；
- R2 adapter≤1M参数、冻结encoder，20 epochs sanity；
- 过闸门后100 epochs×3 seeds。第二轮只对少量refine query，单独报告平均decoder calls。

## 5. 执行顺序

```text
M0/M1 (oracle one-vs-multi-query, prompt form)
        ↓
T0 (target-only vs correct/shuffled background)
        ↓
R0/R1 (response AUPRC and exact candidate credit)
        ↓
只选择通过信号最强的一个：M2 或 T1/T2 或 R2
        ↓
20-epoch sanity → 100-epoch 3-seed screening → 第二数据集
```

M/T/R的零训练诊断可复用同一candidate与image-embedding cache，但不得把validation GT用于部署输入。选择主线后，另外两条只作为反事实/消融，不做三个长训练并行堆模块。

## 6. 预注册结论语言

- 只有M0/M2通过：可说“candidate isolation/no-object解决了point-set污染”，不能说首次object query SAM。
- 只有T0/T2通过：可说“candidate-local background condition在tiny IR中被decoder因果消费”，不能说首次foreground/background prompt。
- 只有R0/R2通过：可说“mask response可预测并拒绝harmful candidates”，不能说首次response adaptation。
- 三者都不通过：停止Self-Prompt主创新，回到image encoder/domain response适配或非SAM强baseline。
