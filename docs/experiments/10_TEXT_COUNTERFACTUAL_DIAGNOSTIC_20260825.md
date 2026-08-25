# E4：文本反事实 `C/N/S/W/O` 确定性诊断

执行日期：2026-08-25。该实验用于判断 E3 的 presence/count role token 是否真正向最终 mask 提供了与图像对应的语义增量，以及 Prompt→Mask 行为蒸馏是否具备 teacher 前提。

## 1. 评测契约

- 模型：E3 的验证集 mIoU 最佳 checkpoint；IRSTD-1k 为 epoch 768，NUAA-SIRST 为 epoch 991；
- 图像预处理：SCTransNet 归一化，固定 `resize` 到 256×256，不使用随机验证裁剪；
- 图像编码：同一 batch 的五种条件共享一次 image encoder forward；
- SAM 提示：`prompt_mode=assp_only`，`point_coords=[B,1,0,2]`，`point_labels=[B,1,0]`，box 与 mask prompt 均为空；
- 文本接口：保持 E3 的双 role token projector 与模型容量不变；
- 阈值：先在 `N` 条件上按 mean IoU 选择一个固定阈值，再将该阈值应用到全部条件。该阈值只用于机制诊断，不作为独立 test-set 最终报告；
- `C/N/S/W` 构造过程不接收评测 GT。仅 `O` 从 GT mask 提取 presence/count，并明确标记为 `ORACLE / NOT DEPLOYABLE`。

条件定义：

| 条件 | 输入 | 是否读取评测 GT | 作用 |
| --- | --- | ---: | --- |
| C | 当前图像缓存的 GPT presence/count role tokens | 否 | 正确/原始文本条件 |
| N | token feature 与 attention mask 全部清零，但保留同一个 learned projector | 否 | 匹配容量的 null-text 条件 |
| S | 固定 seed 的无自匹配乱序文本 | 否 | 检查图文对应关系 |
| W | 只根据缓存 GPT 值反转 presence，并令 count 为 0/1 | 否 | 错误语义压力测试 |
| O | GT mask 的目标存在性与连通域数量 | 是 | 不可部署的文本理论上界 |

## 2. 固定 `N` 阈值结果

表中 IoU、F1、Pd、AUPRC 均为百分数；Fa 单位为 `×10^-6`。`global IoU` 是数据集累计 intersection/union，`mean IoU` 是逐图 IoU 均值。

### 2.1 IRSTD-1k（201 张，固定阈值 0.95）

| 条件 | global IoU | mean IoU | F1 | Pd | Fa | mask AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C | 61.28 | 51.71 | 63.78 | 84.88 | 22.39 | 74.36 |
| N | 60.04 | 51.81 | 64.01 | 86.25 | 28.16 | 72.34 |
| S | 61.26 | 51.69 | 63.77 | 84.88 | 22.39 | 74.37 |
| W | 61.28 | 51.71 | 63.78 | 84.88 | 22.39 | 74.36 |
| O | 61.30 | 51.72 | 63.79 | 84.88 | 22.24 | 74.36 |

`C-N` 的逐图 mean-IoU 差值为 -0.10 个百分点，标准误为 0.57 个百分点，2000 次 bootstrap 95% CI 为 `[-1.19, +0.99]` 个百分点。CI 跨 0，不能认为正确文本优于 no-text。

### 2.2 NUAA-SIRST（214 张，固定阈值 0.25）

| 条件 | global IoU | mean IoU | F1 | Pd | Fa | mask AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C | 66.51 | 62.93 | 75.76 | 92.78 | 3.57 | 78.09 |
| N | 66.49 | 62.91 | 75.74 | 92.78 | 3.57 | 78.08 |
| S | 66.51 | 62.93 | 75.76 | 92.78 | 3.57 | 78.09 |
| W | 66.51 | 62.93 | 75.76 | 92.78 | 3.57 | 78.09 |
| O | 66.51 | 62.93 | 75.76 | 92.78 | 3.57 | 78.09 |

`C-N` 的逐图 mean-IoU 差值为 +0.02 个百分点，标准误为 0.02 个百分点，2000 次 bootstrap 95% CI 为 `[-0.02, +0.07]` 个百分点。CI 跨 0，且只有 1.40% 图像的逐图 IoU 严格提高。

## 3. 语义敏感性诊断

不同文本确实经过 learned projector 形成了不同 sparse embeddings，但这些差异几乎没有传递到最终概率图：

| 数据集 | 对照 | sparse prompt mean L2 | 概率图 mean absolute difference |
| --- | --- | ---: | ---: |
| IRSTD-1k | S vs C | 0.8571 | `8.08×10^-8` |
| IRSTD-1k | W vs C | 1.6152 | `8.23×10^-8` |
| IRSTD-1k | O vs C | 0.2999 | `3.69×10^-8` |
| NUAA-SIRST | S vs C | 0.1076 | `8.49×10^-10` |
| NUAA-SIRST | W vs C | 0.7653 | `4.32×10^-9` |
| NUAA-SIRST | O vs C | 0.0352 | `3.19×10^-10` |

因此，当前模型主要响应“是否存在活跃 role tokens”造成的整体校准变化，而没有表现出可靠的文本内容或图文对应敏感性。`C≈S≈W≈O` 也说明继续提高 GPT 描述准确率不能直接修复该机制。

## 4. 决策与停止条件

两个筛选数据集均未证明 `C` 相对 `N` 存在稳定正增量，且 oracle 文本没有形成可辨识的额外 mask 行为。根据预注册条件：

1. 暂停 `B4/B5` 文本反事实增量蒸馏，不训练一个学生去复制当前 teacher；
2. 不再把“presence/count role token 直接作为 SAM sparse prompt”作为主线；
3. 先运行相同容量的 null-token 训练探针，判断此前差异是否只是 token 激活与阈值校准效应；
4. 主线随后转向高分辨率视觉 self-prompt，文本仅在证明能验证候选或降低 Fa 时作为可拒绝的可选 gate。

## 5. 可复现产物

- 评测脚本：`scripts/eval_text_counterfactuals.py`；
- GT 边界单元测试：`tests/test_eval_text_counterfactuals.py`；
- 本地结果：`outputs/text_counterfactual_eval_20260825/{IRSTD-1k,NUAA-SIRST}`；
- 每个数据集均保存 `manifest.json`、`summary.json`、`aggregate_metrics.csv`、`per_image_metrics.csv` 和 `paired_deltas_vs_N.csv`；
- IRSTD-1k checkpoint SHA-256：`c840e68812da84935d4a6491c2900dc53d554a1547096b11b412a3673c6cd91c`；
- NUAA-SIRST checkpoint SHA-256：`d37818feac843c05619ded56ca8a8fe1608398019dc9f8cf1cc620d20cc576ce`；
- 两个测试集的 C 条件均为全部样本启用 2 个 role tokens；
- 在线生成控制文本与原缓存 CLIP token 的最大绝对误差为 `2.24×10^-4`。

## 6. 已完成的匹配 no-text 探针

2026-08-25 11:41 CST 在本地 RTX 5090 D 上启动 IRSTD-1k 100-epoch 筛选，12:35 CST 完整结束：

- 仍使用 2-token `fused_tokens` projector，参数量和 E3 相同；
- 输入 cache 的 `token_features/global_feat/attention_mask` 全部为 0；
- 从同一个 `weights/efficient_sam_vitt.pt` baseline 初始化；
- 训练设置保持 256×256、batch size 4、HQ warm-up 30 epochs、encoder freeze 60 epochs；
- 验证固定使用 `sc_eval_crop=resize`；
- 输出：`outputs/model1_matched_null_probe_20260825`；
- 日志：`job_logs/model1_matched_null_probe_20260825`；
- null cache SHA-256：`cad968f7e3f7c8171a8eedd07be102cad5ef4860e9a4cd1544cb66e9a6b4c34a`。

训练日志最佳 checkpoint 为 epoch 95：global IoU=59.22、mean IoU=49.46、F1=61.66、Pd=83.85、Fa=23.61，阈值为 0.41。epoch 100 的 global IoU=57.79、mean IoU=49.76、F1=62.01、Pd=89.00、Fa=39.00。stderr 为空，无 NaN、Inf、OOM、RuntimeError 或 Traceback。

使用同一 E4 评测器、固定 `resize` 和各自的数据集级最优固定阈值，对比 100-epoch 范围内保存的 E3 epoch 97 与 matched-null epoch 95：

| 探针 | 条件 | 阈值 | global IoU | mean IoU | F1 | Pd | Fa | mask AUPRC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E3 role token, epoch 97 | C | 0.95 | 55.47 | 50.01 | 61.96 | 85.22 | 54.05 | 69.98 |
| matched null token, epoch 95 | N | 0.30 | 58.17 | 48.93 | 61.23 | 84.88 | 27.41 | 72.77 |

E3-100 的 mean IoU 比 null-token 高 1.08 个百分点，但 global IoU 低 2.70 个百分点、Fa 高 26.65、AUPRC 低 2.79 个百分点，是混合胜负。另外，E3 checkpoint 依随机验证裁剪保存，matched-null 依固定 `resize` 保存，当前只有单种子，因此该对比只是方向筛选，不是最终三种子结果。

由于 E4 已直接观察到 `C≈S≈W≈O`，且 matched-null 探针没有给出文本模型稳定占优的多指标证据，停止扩展 NUAA-SIRST/null-token 三种子，不启动 B4/B5 行为蒸馏。下一主线转向高分辨率视觉 self-prompt，文本仅作为需单独证明有效的可拒绝候选 gate。
