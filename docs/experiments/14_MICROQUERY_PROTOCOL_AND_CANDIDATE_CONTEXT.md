# MicroQuery-SAM 协议与候选背景条件

更新时间：2026-08-26

## 1. 研究问题与部署边界

本实验把问题严格拆成两层：A1 neck SpatialProbeHead 能覆盖多少真实目标；在完全相同且不完美的候选集合上，独立 query 与候选级审核能否保持已覆盖目标，同时拒绝背景/重复候选并改善最终 mask。

部署前向只接收图像、冻结候选坐标/分数及图像特征。GT 仅出现在训练 loss、匹配和离线指标中。Oracle filter、GT center、GT box 和 GT micro-mask 均标记为 `NOT DEPLOYABLE`，不能进入部署结果。

固定配置如下：

- 基线 commit：`fe24d891181349a240d1b0cc2392a031217b91ed`；
- 分支：`codex/microquery-sam`；
- 数据集：IRSTD-1k；train/val 为 720/80 张；
- seed：20260825；输入固定 resize 256×256；
- candidate source：A1 single-view neck SpatialProbeHead；
- A1-P best-mask SHA-256：`6320c5e2a68aa934b92b869998d826463b630f560f96e4257391deebabc9a904`；
- probe SHA-256：`cc96c90cd19f4b215c535656055e5788b5d9db5cbe2f1fbc2910bb87faa0ae47`；
- val split SHA-256：`3f0206fda5f471690f47570f80990e81059e2282caa8739b04905879c29faa1e`；
- train split SHA-256：`539be5c2c08eeddac03b8c59e57de30fe3d541e2c399163c3cade82124a7d9af`。

## 2. 候选缓存与 GT 隔离

`candidates.npz` 只保存图像名、候选坐标、冻结分数和 valid 位，不包含 GT。GT 派生的 semantic、primary/duplicate/background 和 component id 保存于单独 analysis 文件。M0 首次 smoke 发现重新运行 proposal 时，候选可能因 batch size 的数值差异而不再逐字节相同，因此正式 M0/M2 均直接加载冻结缓存，不重新生成候选。

缓存哈希：

| Split | 图像数 | Candidate cache SHA-256 |
| --- | ---: | --- |
| train | 720 | `9d36faa138915023cb265523483202317d81069096519e1d8c01c14504b72ffd` |
| val | 80 | `4269eb06c7f7d7dcec1edc55cdf4676bca4da700e95c58ad2963db5451766230` |

M2 另生成部署特征 `features.npz` 与 GT 分析标签 `analysis_targets.npz`。部署特征由 shallow ROIAlign 7×7、neck ROIAlign 3×3 的池化特征、归一化坐标和冻结候选分数组成，共 451 维；GT 不参与 ROI 提取。

## 3. Candidate Coverage@K

正式结果来自 A1-P best-mask encoder，而不是早期只评 probe 的旧缓存，因此数值与旧 P0 报告不同。validation 共 117 个 GT components。

| K | Coverage@K | All-target image coverage | tiny 1–9 px | 10–16 px | 17–25 px | >25 px | duplicate/component | background/image |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 94.02% | 93.75% | 88.14% | 100% | 100% | 100% | 0.0598 | 1.9000 |
| 10 | 95.73% | 96.25% | 91.53% | 100% | 100% | 100% | 0.0769 | 2.7875 |
| 20 | 95.73% | 96.25% | 91.53% | 100% | 100% | 100% | 0.0769 | 3.7625 |

train 的 Coverage@5/10/20 分别为 91.88%/92.07%/92.44%，说明 K=10 已覆盖绝大多数可覆盖目标，同时仍保留足够背景负例用于 M2-S1。

## 4. 评价协议

所有 M0/M2 对照固定候选坐标、顺序、分数、checkpoint、split、K 与 mask decoder。正式主指标为 Coverage、CTR、TCR、FCRR、DSR、global IoU、mean nIoU、Pd、Fa 和 mask AUPRC。mask threshold 扫描 0.05–0.95；M0 使用 2000 次配对 bootstrap。M2-S1 的 object threshold 先要求 TCR≥99.5%，再最大化背景/重复候选拒绝；若无阈值满足，选择 TCR 最高的诊断点并明确判为未过门控。

输出位于 `outputs/microquery/`，由 `.gitignore` 排除，不提交大型概率图、checkpoint 或数据缓存。

