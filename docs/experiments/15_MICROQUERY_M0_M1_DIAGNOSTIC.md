# MicroQuery-SAM M0/M1 诊断结果

更新时间：2026-08-26

## 1. M0：固定预测候选的 one-query 与 independent-query

下表为 IRSTD-1k validation、mask threshold=0.5。`M0-Micro-max` 对每个候选独立解码后逐像素 max；`Oracle filter` 使用 GT 删除背景/重复候选，仅作不可部署上界。

| K | 条件 | CTR | TCR | FCRR | global IoU | mean nIoU | Pd | Fa ×10⁻⁶ | 结论 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 5 | M0-One | 92.73% | 100% | 0% | 55.61% | 48.13% | 88.03% | 44.06 | 同候选基线 |
| 5 | M0-Micro-max | 93.64% | 100% | 0% | 50.48% | 46.98% | 89.74% | 88.88 | CTR 略升但 Fa +101.7% |
| 5 | Oracle filter | 93.64% | 100% | 100% | 55.13% | 50.07% | 88.03% | 33.38 | 不可部署上界成立 |
| 10 | M0-One | 89.29% | 100% | 0% | 53.15% | 46.14% | 86.32% | 53.60 | 同候选基线 |
| 10 | M0-Micro-max | 92.86% | 100% | 0% | 50.31% | 46.76% | 89.74% | 90.98 | CTR +3.57pp，但 Fa +69.8% |
| 10 | Oracle filter | 92.86% | 100% | 100% | 55.22% | 50.65% | 88.89% | 35.29 | 不可部署上界成立 |
| 20 | M0-One | 90.18% | 100% | 0% | 53.76% | 47.59% | 87.18% | 49.40 | 同候选基线 |
| 20 | M0-Micro-max | 92.86% | 100% | 0% | 50.29% | 46.74% | 89.74% | 91.17 | CTR +2.68pp，但 Fa +84.6% |

independent query 的 Best Query Mask IoU 在 K=10 从 one-query 分析值 0.3395 提高到 0.3851，且 correct coordinate、batch-shuffled coordinate、invalid label 的输出不相同，说明 decoder 确实响应候选位置。问题在于未审核的背景 query mask 一并进入 max 聚合，直接放大虚警。

Top-1 可在 K=10 将 Fa 降到 43.68×10⁻⁶，但 TCR 仅 64.29%，属于以误拒真候选换虚警，不能通过。GT Oracle filter 在保持 TCR=100% 时明显降低 Fa，证明候选级拒绝存在可利用上界。没有任何可部署 M0 聚合满足门控，因此 M0 未通过。

## 2. M1：Oracle prompt 表示上界

以下全部为 `NOT DEPLOYABLE`。每个 GT component 单独提供 prompt；mask threshold=0.5。

| 条件 | CTR/Pd | global IoU | mean nIoU | Fa ×10⁻⁶ | Best Query IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Null | 76.07% | 55.38% | 44.81% | 45.39 | 0.3287 |
| GT point one-query | 89.74% | 55.29% | 51.07% | 34.71 | 0.3615 |
| GT point independent | 90.60% | 55.31% | 51.38% | 36.05 | 0.3784 |
| GT point + negative points | 87.18% | 57.08% | 50.38% | 29.95 | 0.3702 |
| GT box | 76.92% | 55.77% | 44.91% | 46.16 | 0.3241 |
| GT micro-mask | 76.07% | 55.85% | 44.11% | 23.84 | 0.3208 |
| GT point + micro-mask | 87.18% | 56.76% | 49.03% | 28.80 | 0.3596 |
| GT box + micro-mask | 76.07% | 55.82% | 44.44% | 48.26 | 0.3214 |

GT point 相对 Null 将 Pd 提高 14.53pp、mean nIoU 提高 6.57pp，说明 point prompt 接口有效。independent GT point 相对 one-query 只增加 0.85pp Pd 和 0.31pp mean nIoU，候选隔离本身不是充分机制。

micro-mask 启动闸门未通过：tiny Best Query IoU 相对 point 为 −4.13pp，tiny component detection 为 −6.78pp，总体 mean nIoU 为 −2.35pp。因此 M2 不实现 micro-mask head。M0 虽失败，但 M1 point 有效且 Oracle filter 有效，按 V2 规则只继续有限的 M2 objectness 诊断。

