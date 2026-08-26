# MicroQuery-SAM M3 最终验证状态

更新时间：2026-08-26

## 1. 当前状态

M3 未启动，这不是缺失结果，而是执行预注册停止规则。M2-S1 虽将 semantic AUPRC 提高 10.29pp、Fa 降低 51.78%、CTR 保持不变，但 TCR 下降 4.46pp，超过允许的 0.5pp；M1 又已否决 micro-mask。因此没有冻结的、满足机制安全门槛的 M2 结构可进入 100-epoch、三随机种子与跨数据集验证。

以下项目均标记为“因上游门控失败而停止”，没有生成或虚构数值：

| 阶段 | 计划 | 状态 | 停止原因 |
| --- | --- | --- | --- |
| M2-S2 | IRSTD-1k 100 epochs single seed | 未启动 | M2-2 TCR 门控失败 |
| M2-S3 | 3 seeds × 100 epochs | 未启动 | 无合格单种子结构 |
| M3-NUAA | 第二数据集机制验证 | 未启动 | 等级 A 尚未成立 |
| M3-NUDT | 最终泛化确认 | 未启动 | 配置尚不可冻结 |
| hard-negative formal eval | train-derived background set | 未启动 | 当前失败已由有目标 val 上的真候选误拒直接证实 |
| test-set final evaluation | 冻结后只评一次 | 未启动 | 禁止在未过 val 门控时消费 test |

## 2. 已成立与未成立的证据

已成立：独立 query mask 与候选位置有响应；GT point prompt 有效；GT Oracle filter 能在保持 TCR 时降 Fa；轻量 ROI objectness 明显优于 raw candidate score；correct/zero/shuffled/wrong 反事实不同。

未成立：可部署 M0 聚合；component-safe no-object；micro-mask；quality weighted aggregation；两个 validation 数据集、3 seeds 与 Pd–Fa Pareto 泛化。因此 MicroQuery 目前只能作为诊断方向，不能作为论文主模型组成。

## 3. 若以后重启的最小条件

只有新的拒绝机制在同一冻结候选上满足 `Fa下降≥10%、CTR下降≤0.5pp、TCR下降≤0.5pp、objectness AUPRC较raw至少+3pp`，才允许重启 M2-S2。建议优先研究每个候选的保守 abstention/分组保留或无需 GT 的覆盖约束，而不是延长当前二分类头训练；重启时仍必须重新从 20-epoch sanity 过门控。

