# MicroQuery-SAM M2-S1 结果与停止决策

更新时间：2026-08-26

## 1. 最小原型

M2-S1 固定 K=10，只训练 183,177 个参数：451 维候选 ROI descriptor 经两层 MLP 投影到 256 维，再输出 target/no-object logits 与 quality logit。训练 20 epochs、seed 20260825；image encoder、candidate probe、SAM prompt encoder 和 mask decoder 全部冻结。best epoch=18，选择依据为 validation primary AUPRC。

GT 边界已物理隔离：`features.npz` 不含 GT；`analysis_targets.npz` 单独保存 primary/duplicate/background、query IoU 和 GT mask。部署 `MicroQueryHead.forward` 只接收 descriptor 与 candidate validity。

## 2. Objectness 结果

| 指标 | Raw candidate score | M2 objectness | 增量 |
| --- | ---: | ---: | ---: |
| Semantic AUPRC | 82.53% | 92.82% | +10.29pp |
| Primary AUPRC | 78.54% | 91.94% | +13.40pp |
| Semantic AUROC | 85.67% | 94.94% | +9.27pp |
| Semantic ECE | 46.27% | 5.29% | −40.98pp |
| Semantic Brier | 0.4457 | 0.0850 | −0.3607 |

分类头本身明显优于 frozen candidate score，满足 AUPRC 至少 +3pp 的条件。

## 3. 同候选最终 mask

object threshold 扫描 0.05–0.95。没有阈值达到 TCR≥99.5%，因此按预注册 fallback 选择 TCR 最高、背景拒绝最强的诊断点 0.15。下表固定 mask threshold=0.5；M2-1 与 M2-2 使用相同 K=10 query masks。

| 条件 | CTR | TCR | FCRR | DSR | global IoU | mean nIoU | Pd | Fa ×10⁻⁶ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M2-1 independent all | 92.86% | 100% | 0% | 0% | 50.34% | 46.79% | 89.74% | 90.98 |
| M2-2 object hard gate | 92.86% | 95.54% | 83.86% | 44.44% | 54.37% | 50.25% | 88.89% | 43.87 |
| M2-2 object weighted | 89.29% | 100%* | 0%* | 0%* | 43.40% | 47.41% | 85.47% | 35.10 |
| object × quality weighted | 7.14% | 100%* | 0%* | 0%* | 3.71% | 3.25% | 6.84% | 0.00 |

带 `*` 的 weighted 条件没有做 hard rejection，TCR/FCRR/DSR 只反映 valid query 集合，不能解释为真正保留/拒绝。quality 权重在固定 0.5 下把概率整体压低，不能保留为有效聚合。阈值扫描中各条件的最佳 global IoU 已保存，但扫描无法修复 hard gate 的 TCR 损失，因为 TCR 由 object threshold 决定。

M2-2 hard gate 相对 M2-1：

- semantic AUPRC +10.29pp；
- Fa −51.78%；
- CTR +0.00pp；
- global IoU +4.03pp、mean nIoU +3.46pp；
- 但 TCR −4.46pp，Pd −0.85pp。

因此这是“背景审核能力成立，但真候选误拒过多”的混合结果。它不能按 V2 第 9.2 节称为 M2-2 有效，最终 `m2_2_gate_passed=false`。

## 4. 反事实

所有反事实保持候选坐标、独立 query masks 和模型容量不变，只修改 descriptor/objectness 对应关系；固定 object threshold=0.15、mask threshold=0.5。

| 条件 | CTR | TCR | global IoU | Pd | Fa ×10⁻⁶ |
| --- | ---: | ---: | ---: | ---: | ---: |
| correct objectness | 92.86% | 95.54% | 54.37% | 88.89% | 43.87 |
| zero descriptor | 92.86% | 100% | 50.34% | 89.74% | 90.98 |
| batch-shuffled descriptor | 85.71% | 69.64% | 48.21% | 82.05% | 59.32 |
| wrong/inverted objectness | 71.43% | 33.93% | 42.95% | 69.23% | 89.26 |
| all reject | 0% | 0% | 0% | 0% | 0 |

correct 明显不同于 zero/shuffled/wrong，证明 head 使用了候选内容；但 correct 的 TCR 仍不满足安全门槛。

## 5. 效率与复现

- M2 head：183,177 trainable parameters；
- head MLP batch-80 测得 0.0035 ms/image，峰值 GPU allocation 26.1 MB；该数值不包含 encoder、ROIAlign 和 SAM decoder；
- validation feature cache：encoder 16.04 ms/image、K=10 independent decoder 12.90 ms/image（batch=8）；
- feature cache 峰值显存：3.81 GB（batch=8）；
- 单 image 的 M0 K=20 independent decoder 为 33.38 ms/image，说明 decoder 开销随 K 和 batch 协议变化，不能与 batch-8 数值混用作绝对部署延迟。

核心输出目录：`outputs/microquery/M2_sanity/IRSTD-1k/a1_best_mask_K10_20ep_seed20260825/`。概率图、特征缓存和 checkpoint 不提交 Git。

## 6. 决策

按预注册规则停止 M2-S2 100-epoch、三随机种子和第二数据集长训；不实现 micro-mask、大型 Transformer、GNN、latent candidate token 或 reliability 扩展。当前瓶颈不是 objectness 可分性，而是如何在不使用 GT 的情况下提供“component-safe”拒绝：单候选分类分数高，仍不能保证每个已覆盖 component 至少保留一个 query。

