# Top-3 Idea 筛选实验

目标是用最小可行实验先证伪，不立即进行大规模结构堆叠。本文只定义协议和停止条件，不填写任何未运行结果。

## 0. 统一协议

### 数据、划分与 seeds

- 数据集：IRSTD-1k（800/201）、NUAA-SIRST（213/214）、NUDT-SIRST（663/664），沿用 `50_50/train.txt` 与 `50_50/test.txt` 的现有实际划分。
- 筛选阶段：IRSTD-1k + NUAA-SIRST，3 个 seeds；通过后才在 NUDT-SIRST 确认。NUDT 当前结果接近饱和，更适合验证稳定性/Fa，不适合单独决定 idea。
- 同一比较组必须使用相同初始化、encoder 冻结/解冻日程、输入尺寸、数据增强、decoder、后处理、训练 epoch 与 early-selection 规则。
- 当前 CBGA 跨数据集存在旧 gated 与新 delta-only/bias-free 路径不一致；新实验统一使用稳定的 delta-only/bias-free 版本，不能与旧表直接合并为同一消融。

### 禁止 GT prompt 的部署契约

所有主结果强制：

```text
prompt_mode = assp_only
test point_coords = empty
test boxes = None
test mask_inputs = None
test GT 只进入 metrics，不进入任何 model/prompt/VLM 前处理
```

运行时记录 resolved prompt mode、semantic source、文本条件、checkpoint metadata。若任一 batch 从 GT mask 调用 `sample_points_from_mask`，该 run 作废。GT-derived oracle text/point 只允许在明确标为 `ORACLE / NOT DEPLOYABLE` 的上界实验中出现，且不得进入主表。

### 文本控制组

| 标记 | 输入 | 用途 |
|---|---|---|
| N | 无文本 / zero semantic feature | 强 no-text 下界 |
| F | 固定通用文本，如 “an infrared image with a possible small target” | 检验逐图 caption 是否必要 |
| C | 当前图像的正确 GPT structured caption/CLIP features | 文本 teacher |
| S | batch/数据集内随机打乱的正确 caption | 检验是否真的使用图文对应 |
| W | 人工构造语义错误：目标不存在、相反背景/位置/亮度 | 安全性压力测试 |
| O | 从 GT mask/metadata 渲染的 oracle structured text | 文本质量理论上界；不可部署 |

每个模型至少评估 N/C/S/W；F 与 O 用于定位机制上限。GPT API 仅离线生成 C，推理 benchmark 不包含网络 API。所有 C/S/W 必须在图像进入模型前固定并可复现。

### 最终指标

- 像素：mIoU、nIoU、F1。
- 目标：Pd、Fa（`×10^-6`），并给 95% bootstrap CI。
- 资源：参数量、训练显存、推理显存、单图 latency、是否需要 GPT/CLIP/VLM。
- 面积分层：GT component 面积 `1–9`、`10–16`、`17–25`、`>25` pixels（若 resize 改变，按原图映射后分桶）。

### Prompt 级诊断（先于最终 mask）

| 指标 | 定义 |
|---|---|
| Center-Hit@K | top-K 正点是否落在 GT component 或半径 r 的膨胀区 |
| Component Recall@K | 被至少一个 accepted point/dense component 覆盖的 GT components 比例 |
| Prompt Precision | accepted prompt components 中与任一 GT 相交者比例 |
| False Prompts/MP | 每百万像素落在背景且被接受的 prompt 数 |
| Dense Prompt AUPRC | prompt map 对 GT mask 的像素 AUPRC；避免只看阈值后结果 |
| Presence AUROC/AUPRC | 图像是否含 target 的校准能力；需构造/保留 empty/hard-negative 样本 |
| ECE / Brier | reliability/presence 概率校准 |
| Risk–Coverage | 随 abstention 提高，保留样本上的 prompt error 与覆盖率关系 |

---

## Experiment 1：可靠性校准的多视图 Prompt 与拒绝机制

### 核心问题

自动 prompt 在多视图上一致时是否更可信？显式拒绝是否能在保持 tiny-target recall 的同时降低 false prompts 和 Fa？

### 最小实现

- 视图：原图、两档尺度、局部对比增强、一个轻量高频残差；先离线/共享 encoder，避免一次加入昂贵多分支。
- 每个 view 输出 targetness，逆变换回原坐标；计算均值、方差、component center dispersion。
- 先使用无学习规则 gate（均值高且方差低），  证明信号存在；再训练轻量 ReliabilityHead。
- 文本 C 只在训练期给 candidate soft label/presence，不直接生成坐标；最终 student 只看图像多视图统计。

### 消融阶梯

| 编号 | Prompt 生成 | Reliability | 文本 | 推理依赖 |
|---|---|---|---|---|
| A0 | 当前 global ASSP | 无 | N | 图像 |
| A1 | 单视图同容量 targetness | 无 | N | 图像 |
| A2 | 多视图 targetness | 无 | N | 图像 |
| A3 | A2 + rule consistency gate | 固定 | N | 图像 |
| A4 | A2 + learned gate | 学习 | N | 图像 |
| A5-T | A4 + text candidate verifier | teacher | C/F/S/W/O | 图像+离线文本 |
| A5-S | 蒸馏 verifier/gate | student | N | 纯图像 |

容量控制：给 A1 增加与多视图 head 等量参数；给普通 ensemble 同样 view 数，排除“多算几次”因素。

### 必做压力测试

- C/S/W/N 下分别画 risk–coverage 与 Pd–Fa 曲线。
- 空目标/纯背景 crop、云边/建筑灯/强边缘等 hard negatives。
- 按面积桶统计 Component Recall，特别检查 1–9 pixels 是否被 variance gate 错删。
- gate 阈值只在验证集选择，不按 test GT 调整。

### 预注册通过/停止条件

进入完整训练需同时满足两个筛选数据集、至少 2/3 seeds：

- A3/A4 相对 A2 的 False Prompts/MP 或 Fa 至少降低 10%，同时 Component Recall@K/Pd 的绝对下降不超过 0.5 个百分点；
- A5-S 在 C teacher 增益可见时，保留 A5-T 至少 70% 的 Fa 改善；
- tiny `1–9` 桶 recall 不低于 A0。

若 rule gate 已无信号、A5-T 不优于 A4、或 reliability 不优于 max-score baseline，则停止 Idea 1，不训练复杂 gate。

---

## Experiment 2：文本反事实增量的 Prompt→Mask 行为蒸馏

### 核心问题

teacher 的正确文本是否真的带来相对 no-text 的有价值增量？若有，只蒸馏该反事实增量及 wrong/shuffle 的安全边界，是否优于现有 embedding regression TASSG 和普通无条件 mask/logit KD？

### Phase B0：先验证 teacher 可蒸馏

同一 checkpoint 与图像分别运行 N/F/C/S/W/O，记录：

- `Δmask(C,N)` 的 IoU/Pd/Fa 改变；
- sparse prompt cosine/set distance、dense prompt AUPRC 与 component ranking；
- C 相对 S/W 是否稳定更好；O 是否仍有上升空间。

若 C 相对 N 的平均改善不超过 3 seeds 的配对标准误，或 C 与 S/W 无差异，则停止：没有证据表明 GPT teacher 含有可蒸馏的新行为。

### 蒸馏阶梯

| 编号 | Student loss | 蒸馏对象 | 推理文本 |
|---|---|---|---|
| B0 | mask GT only | 强 no-text baseline | 无 |
| B1 | B0 + embedding cosine/MSE | CLIP global/token（现有 TASSG） | 无 |
| B2 | B0 + `L_prompt` | sparse set + dense prompt | 无 |
| B3 | B2 + `L_mask` | teacher mask logits / component scores | 无 |
| B4 | B3 + `L_delta` | C-N 的条件行为增量 | 无 |
| B5 | B4 + perturbation consistency | prompt→mask 有限差分响应 | 无 |

另加 `B3-plainKD`：同容量普通 teacher mask/logit KD。它用于排除 EdgeSAM、SAM-COD、MS-SAM-LESS 一类成熟 prompt/mask 蒸馏范式即可解释增益；B4 必须优于该控制，才能支持“文本反事实增量”的必要性。

Teacher 固定，所有 soft targets 离线缓存。B1–B5 使用相同 student 参数量；额外 projector 参数需给 B0 的容量对照。

### 反事实设计

- teacher 对 C/S/W/N 产生四套 outputs；student 不直接复现 W 错误，而学习 `C-N` 正增量和 `W≈N` 安全边界。
- `L_delta` 只在 teacher C 确实优于 N 且置信高的样本生效，防止蒸馏 teacher false alarm。
- 另报 teacher upper bound、student absolute score 和“保留 teacher 增益比例”。

### 预注册通过/停止条件

- B3/B4 必须在两个筛选数据集平均优于 B1：mIoU 或 nIoU 至少 +0.5 个百分点，且 Fa 不恶化超过 5%；
- 至少一个 tiny-area prompt recall 指标和一个最终指标同时改善，避免仅拟合 logits 温度；
- 跨数据集 direct transfer 不得比 B1 更差超过 1 个百分点。

若 B1≈B4、teacher C≈N、或 student 主要复制 teacher 的 W/S 失效，则停止 Idea 2。B5 只有 B4 通过后才做。

---

## Experiment 3：高分辨率残差语义定位

### 核心问题

当前失败主要来自低分辨率空间证据，还是 caption/融合不够强？局部高分辨率 residual 负责定位、global semantics 只校准背景，是否优于文本直接投影 prompt？

### 首轮不训练诊断

从当前 checkpoint 抽取 1/4、1/8、1/16 features：

- 对每个 GT component 测 feature activation/similarity 的 Center-Hit 与 AUPRC；
- C/N/S 文本条件下比较各层变化；
- 若浅层可见、深层消失，支持 high-resolution path；若所有层均不可见，先修视觉 encoder；若深层已有高 recall，Idea 3 不成立。

### 结构消融

| 编号 | 定位证据 | Semantic 作用 | 目的 |
|---|---|---|---|
| C0 | 当前 global ASSP | 当前 text/no-text | 基线 |
| C1 | 普通同参数 FPN/upsampler | 无 | 排除容量/上采样收益 |
| C2 | shallow high-res feature | 无 | self-prompt baseline |
| C3 | local contrast/frequency residual + shallow | 无 | 验证 IR-specific evidence |
| C4 | C3 + global bias subtraction | image global | 验证背景原型 |
| C5-T | C4 + text semantic calibration | C/F/S/W/O | teacher |
| C5-S | C4 + image semantic student | N | 纯图像部署 |

所有模型将定位 map 统一转为同尺寸 dense prompt，再送入相同 SAM decoder；不允许 C5 增加额外 decoder 层来混淆来源。

### 关键控制

- `C5-T(C)` vs `C5-T(S/W)`：若无差别，文本 gate 没在使用语义。
- C3 vs C1：判断 residual 机制是否超出普通 FPN。
- C4 vs C3：判断 global bias subtraction 是否只调阈值。
- C5-S vs C5-T：量化纯图像部署保留比例。
- 画每个面积桶 prompt map AUPRC/recall，而非只报告最终 IoU。

### 预注册通过/停止条件

- C3 相对 C1 在 `1–9` 与 `10–16` 桶 Component Recall@K 均至少 +3 个百分点，且 False Prompts/MP 不升高超过 10%；
- C5-T(C) 相对 C4 主要降低 false prompts/Fa，并且 C5-T(S/W) 不应优于 C；
- C5-S 至少保持 C5-T 增益的 70%，否则无法回答无文本部署。

若普通 FPN=C3、oracle text O 仍不改善、或 high-res map 提高 recall 但 Fa 不可控制，则停止 Idea 3 或把它降级为视觉 baseline。

---

## 4. 执行顺序与资源闸门

1. **只读诊断**：teacher C/N/S/W/O 差异；多层 prompt recall；不训练新模型。
2. **低成本 probe**：Idea 1 rule gate、Idea 3 C1–C3、Idea 2 B0/B1/B2，各 50–100 epochs 或固定小样本，观察是否越过停止线。
3. **单数据集完整筛选**：先 IRSTD-1k，3 seeds；通过才加 NUAA-SIRST。
4. **确认实验**：统一架构在 NUDT-SIRST 运行；最后才考虑组合 Idea 1+2 或 1+3。

禁止在单个 run 不佳后同时加入 CBGA、ASSP、high-res、gate、多个蒸馏 loss。每一阶段只回答一个机制问题，并保留命令、git commit、seed、resolved args、日志和 best-checkpoint 选择规则。

## 5. 结果表模板

| 方法 | Text test | VLM test | GT prompt test | mIoU | nIoU | F1 | Pd | Fa | CompRecall@K tiny | FalsePrompt/MP | ECE | Params/FLOPs |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline `assp_only` | No | No | No | — | — | — | — | — | — | — | — | — |
| Teacher-C | Yes | Offline | No | — | — | — | — | — | — | — | — | — |
| Teacher-S/W | Yes | Offline | No | — | — | — | — | — | — | — | — | — |
| Student | No | No | No | — | — | — | — | — | — | — | — | — |

报告均为 mean±std（3 seeds）；不得把 validation-best 与 last epoch 混用，也不得把不同 CBGA 实现的三个数据集拼成同一方法行。
