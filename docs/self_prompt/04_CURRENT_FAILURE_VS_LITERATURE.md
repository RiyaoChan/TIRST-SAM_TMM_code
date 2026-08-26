# 当前 PR #3 失败机制与文献对照

## 1. 当前真实流程

```text
neck feature
→ single-channel spatial probe
→ five deterministic views
→ inverse warp
→ cluster
→ support/dispersion filtering
→ Top-K all-positive points
→ one SAM query
→ one mask
```

代码证据主要在`efficient_sam/multiview_prompt.py`、`scripts/eval_multiview_prompt_quality.py`与`scripts/eval_experiment1_masks.py`。`CandidateCluster`确实保留support、center dispersion、score variance和local contrast；`clusters_to_proposal`却把筛后的候选重新压成坐标/score，最终所有有效候选仍作为正点进入同一个SAM query。reliability没有成为prompt token、dense state、no-object logit或mask aggregation weight。

## 2. 已有实验直接说明了什么

IRSTD-1k新validation（80张，seed 20260825，固定resize）中：

| 现象 | Prompt-level | Mask-level |
|---|---|---|
| A2 五视图 | Recall@20 94.02%，tiny 89.83%，False Prompts/MP 256.16 | global IoU 56.09，Pd 86.32，Fa 33.57 |
| A3 rule gate | Recall@20 97.44%，tiny 94.92%，False Prompts/MP 223.16，candidate AUPRC 69.92% | global IoU 55.58，Pd 87.18，Fa 40.44 |
| A3相对A2 | prompt recall +3.42pp，false prompts -12.88% | IoU -0.52pp，Fa +20.45% |
| A0 null prompt | — | global IoU 56.02，Pd 88.89，Fa 50.54 |

关键结论不是“A3 gate不好”，而是**更好的候选排序/筛选没有稳定映射成更好的mask**。A2相对A0的IoU只高0.07pp且Pd低2.56pp；A3甚至把prompt-level改善转成了更差的global IoU/Fa。这与SPARK-SAM定义的prompt–response gap高度一致。

## 3. 逐环节失败审计

| 当前环节 | 观察到的结构限制 | 文献中的更强结构 | 当前应做的判别实验 |
|---|---|---|---|
| neck feature | 低分辨率、encoder/probe可能发生feature drift；tiny目标到neck已不可见 | Sam2Rad多层PPN、SAM-SPL浅层dense prompts、De-LightSAM pre-mask、AutoPrompt-SAM3D tri-layer | 固定checkpoint测early/mid/neck component recall和oracle micro-mask上界 |
| single-channel probe | 只表示targetness，不能表示对象identity、background、no-object、shape | MaskSAM/RSPrompter object queries；IP-SAM paired prompt；Sam2Rad box+mask+latent | 同容量`K` query+micro-mask vs 单通道heatmap |
| five views | 只增加观测次数，不增加prompt语义/对象结构；延迟约4× | EviPrompt evidential belief；TEP-SAM temporal query（但协议不同） | 固定同计算预算的ensemble control；停止新增视图 |
| inverse warp/cluster | cluster是几何后处理，不能学到candidate-to-mask因果贡献 | GPRN graph prompt、PromptPilot marginal credit | leave-one-candidate-out mask delta；若无信号，不训练graph/MLP |
| support/dispersion gate | reliability只用于keep/rank；被选后身份消失 | SPARK-SAM token gate；IP-SAM background gate；MaskSAM class/no-object | reliability置零/随机/shuffle并测mask变化；若mask不变则decoder未消费 |
| Top-K all-positive | false candidate不能作为negative evidence；多点可能被SAM解释为同一对象的多个click | EviPrompt正负点；IP-SAM前/背景state；Memory-SAM前/背景retrieval | target-only vs target+outer-ring background vs shuffled background |
| one SAM query | 多个目标、重复候选和错误候选共享attention与一个mask，无法归因/拒绝 | MaskSAM、RSPrompter一query一mask+`∅` | `[B,1,K,2]` vs `[B,K,1,2]`，并逐mask objectness/IoU加权 |
| one mask | 只可全局阈值，不能在candidate级拒绝；错误点污染整张mask | per-query micro-mask + no-object aggregation | max、objectness weighted、reliability×SAM-IoU aggregation |

## 4. 必须回答的十一项问题

### 4.1 Prompt 表示是否过弱？——是

当前一个点只表达`(x,y,positive)`，没有候选外观、尺度、局部shape、背景context和存在性。Sam2Rad与MaskSAM说明同一candidate至少可以联合携带`center/box + micro-mask + latent token + objectness`。单通道热图不应继续作为主表示。

### 4.2 是否缺 no-object？——是

当前Top-K选择后没有`∅`类别，最多只能在采点前过滤。MaskSAM/RSPrompter把unmatched query训练为no-object；这允许“候选进入decoder后被拒绝”，与pre-filter本质不同。

### 4.3 是否缺 background prompt？——是

所有候选标签为正。IP-SAM证明背景条件经过冻结prompt encoder再非对称抑制优于feature addition；EviPrompt证明正负证据可分开建模。IRSTD的outer ring/global background prototype应成为独立state，而不是简单拼feature。

### 4.4 是否缺 candidate query？——是

cluster只是一组坐标记录，进入SAM后没有candidate identity。没有query就无法Hungarian匹配、duplicate penalty、per-candidate objectness或leave-one-out credit。

### 4.5 是否缺 mask feedback？——是

当前SAM mask只用于最终评价。AoP-SAM用mask overlap去冗余，AlignSAM/PromptPilot用response学习动作，H-SAM用stage-1 mask条件stage-2。最小改动不是先上RL，而是离线缓存每候选的独立mask与drop delta。

### 4.6 是否缺 prompt credit？——是

support/dispersion只能说“视图一致”，不能说“这个prompt改善了最终mask”。必须定义`credit_k = quality(M_all) - quality(M_without_k)`或训练期可计算的SAM response proxy，并检验它能否预测有害prompt。

### 4.7 是否缺 decoder response adaptation？——高度可能

A3显著改变prompt-level指标而mask不跟随，是直接证据。SPARK-SAM表明即使oracle target-covering box，冻结SAM2也可能在IR失效；H-SAM/IP-SAM表明decoder/prompt-space适配可能比坐标更关键。应先做frozen-decoder oracle multi-query测试，再决定是否训练adapter。

### 4.8 是否缺 high-resolution micro-mask？——是

点不表达1–25像素目标的局部shape，且neck已下采样。每candidate从early/shallow feature裁出`h×w`micro-mask，既可作为dense prompt，也可作为query-local evidence；这比继续提高heatmap分辨率更有结构性。

### 4.9 是否存在 encoder/probe feature drift？——尚未排除

当前20-epoch probe选择neck的依据是prompt recall综合权衡，不等于100-epoch SAM decoder仍消费同一表征。需冻结image encoder，比较probe checkpoint与mask checkpoint的各层feature中心命中、AUPRC和表示余弦漂移。

### 4.10 reliability 是否在进入 decoder 后消失？——确定是

在`clusters_to_proposal`后，SAM只看坐标/label。support、dispersion与rule reliability只留在auxiliary/日志；两个reliability不同但坐标相同的prompt对decoder完全等价。

### 4.11 多实例是否混在一个 query？——确定是

`[B,1,K,2]`语义是“一个prompt set中的K个点”，不是K个objects。多个真实component、重复点和false prompts共享一个mask token；任何一个错误点都可能改变整张mask，且无法定位责任。

## 5. 与核心 S 级方法的差距

| 方法 | 对象隔离 | 背景/拒绝 | reliability进入decoder | mask反馈 | 对当前失败的直接意义 |
|---|---:|---:|---:|---:|---|
| SPARK-SAM | P | objectness/负环 | Y | response guidance/calibration | 证明IR中的prompt–response gap不能靠坐标修补 |
| IP-SAM | N | **显式背景** | **Y，prompt-space gate** | N | 背景必须作为条件而非普通feature |
| MaskSAM | **Y** | **no-object** | class/query state | auxiliary mask→prompt | 给出完整set prediction模板 |
| Sam2Rad | P | P | box+mask+latent共同消费 | interim mask | 给出轻于Mask2Former的联合prompt state |
| AoP-SAM | N | filter | N | **Y** | response可用于去重/筛点 |
| RSPrompter | **Y** | **C+1** | query token | query mask attention | 证明latent query可绕开物理坐标 |
| SAM-SPL | N | N | dense prompt在encoder/decoder保留 | N | 是必须打败的纯图像IRSTD基线 |
| EviPrompt | N | **正负证据** | standard point | **两轮** | 不确定性/背景比all-positive更合理 |
| PromptPilot | N | 正负/删除 | action Q/credit | **多轮** | 有害prompt suppression已有强近邻 |
| H-SAM | N | attention P | stage state | **两阶段** | decoder response adaptation可能优先于prompt head |

## 6. 因果诊断顺序

```text
Oracle spatial prompt sensitivity
  ├─ oracle multi-query也无收益 → 先做response adapter / prompt grounding
  └─ oracle multi-query有效
       ├─ predicted candidates独立query有效 → 训练object/no-object与aggregation
       └─ predicted candidates仍无效
            ├─ micro-mask oracle有效 → 修high-resolution candidate state
            └─ micro-mask oracle无效 → 修image encoder/domain adaptation
```

对应最低成本反事实：

1. 相同坐标，`reliability=0/1/random/shuffled`；如果mask完全不变，说明现接口没消费可靠性。
2. 相同K点，one-query vs K independent queries；如果oracle独立query也不优，停止object-set长训练。
3. correct/zero/shuffled/wrong/background prompts；如果mask sensitivity近零，停止任何prompt distillation。
4. independent mask的candidate drop；如果drop delta不能区分TP/FP，停止PromptPilot式credit head。
5. oracle center point、oracle tight box、oracle micro-mask；确定真正缺的是坐标、shape还是decoder response。

## 7. 结论

当前实验已经足以**停止**“继续优化多视图heatmap gate”作为主创新。下一步应把self-prompt重新定义为：

> 一个可表示target/background/no-object、保持candidate identity、带高分辨率局部证据、能被decoder验证和拒绝的对象级条件状态。

只有当低成本反事实证明独立query或paired background确实改变mask，才进入训练；否则优先response adaptation，不再扩大prompt generator。
