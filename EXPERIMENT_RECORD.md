# TIRST-SAM 中文实验记录

最后更新：2026-08-12 08:48 CST

本文档只记录已从服务器日志和 checkpoint 名称中核验的实验指标。除非特别说明，表中结果均取验证集 mIoU 最高的 checkpoint，而不是最后一个 epoch。

## 一、评测协议

- 服务器：`202.38.209.226`，NVIDIA RTX 3090 GPU。
- 输入尺寸：256 × 256；常规 batch size 为 4。
- 训练周期：1000 epochs；HQ warm-up 为前 30 个 epochs；图像编码器在前 60 个 epochs 冻结。
- 数据划分：`50_50/train.txt` 和 `50_50/test.txt`。尽管目录名称为 `50_50`，IRSTD-1k 实际包含 800 张训练图像和 201 张测试图像，即预期的 80/20 划分；NUAA-SIRST 为 213/214，NUDT-SIRST 为 663/664。
- 自动提示设置：`prompt_mode=assp_only`。验证阶段和部署推理阶段均不向 SAM 传入任何由 GT 生成的点坐标。
- `use_point_loss` 是使用 mask 监督的训练损失，不属于 SAM 的点提示。
- 除 `Fa` 外，其余指标均以百分数表示；`Fa` 的单位为 `×10^-6`。
- GPT 教师缓存：先将 GPT-5.6 输出的结构化属性转换为确定性文本描述，再使用 CLIP 文本编码器离线编码。

已核验的 GPT/CLIP token 缓存 SHA-256：

| 数据集 | 样本数 | SHA-256 |
| --- | ---: | --- |
| IRSTD-1k | 1001 | `bae23800b7d9539654b1d778b7fffc62eae8270f7c9f47abcc17a2fc3ba34981` |
| NUAA-SIRST | 427 | `c7789dadc0b7002260897a20788b14c1d997dbfcae08fed531a332388cddddcd` |
| NUDT-SIRST | 1327 | `da7f65150ae8ec20b4aa5ee7296b135721fbd8da4229bd28290b4709213fb205` |

## 二、E1：GPT 全局特征输入 ASSP，不使用 GT 点提示

配置：缓存的 GPT/CLIP 全局特征 → 1 个带门控的 `raw_global` ASSP 稀疏 token → SAM 解码器。关闭 token-level CBGA 和旧版 FiLM 文本调制器。

三个数据集均已完成 1000 epochs 训练。

| 数据集 | 最佳 epoch | mIoU | nIoU | F1 | Pd | Fa | 状态 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| IRSTD-1k | 494 | 74.84 | 77.53 | 55.79 | 95.08 | 21.94 | 已完成 |
| NUAA-SIRST | 810 | 75.34 | 77.50 | 83.73 | 96.08 | 27.81 | 已完成 |
| NUDT-SIRST | 870 | 93.08 | 93.90 | 96.50 | 99.37 | 2.21 | 已完成 |

服务器产物：

- 日志：`/home/bip/cry/code/TIRST-SAM_TMM_code/job_logs/model1_gpt5p6_20260810_120128`
- Checkpoint：`/home/bip/cry/code/TIRST-SAM_TMM_code/outputs_model1_gpt5p6_formal`

## 三、E2：GPT token-level CBGA + 全局 ASSP，不使用 GT 点提示

配置：在 E1 基础上，在全部 12 个 ViT blocks 中加入 GPT/CLIP token 特征与视觉 token 之间的门控双向融合。CBGA hidden dimension 为 128，attention heads 为 4，应用间隔为 1，视觉和文本残差系数均为 1.0，gate 初始化 bias 为 -2.0，共增加 2,163,392 个参数。

实验状态（截至 2026-08-12 08:48 CST）：

| 数据集 | CBGA 路径 | 进度 | 最佳 epoch | mIoU | nIoU | F1 | Pd | Fa | 状态 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| IRSTD-1k | 旧版门控路径 | 1000/1000 | 780 | 73.92 | 76.36 | 53.66 | 92.57 | 9.49 | 已完成 |
| NUAA-SIRST | 旧版门控路径 | 1000/1000 | 997 | 77.67 | 78.90 | 82.23 | 95.53 | 12.26 | 已完成 |
| NUDT-SIRST | 旧版门控路径 | 445/1000 | 10 | 25.82 | 45.29 | 56.42 | 89.10 | 795.69 | 无效；模型坍塌后 OOM |
| NUDT-SIRST | 稳定版 delta-only、bias-free 路径 | 678/1000 | 606 | 92.29 | 93.38 | 96.22 | 99.68 | 4.87 | 运行中；临时结果 |

NUDT-SIRST 旧版结果不能作为已完成的 CBGA 实验引用。从第 11 轮到第 445 轮，验证指标完全固定为 `mIoU=22.50`、`nIoU=43.74`、`F1=55.24`、`Pd=91.64`、`Fa=1190 ×10^-6`，表明模型很早就退化为固定输出。第 446 轮在申请 768 MiB 显存时失败，当时进程已经使用 23.16 GiB。因此，OOM 是最终终止错误，并不是性能异常的主要原因。

旧版服务器产物：

- 日志：`/home/bip/cry/code/TIRST-SAM_TMM_code/job_logs/model1_gpt5p6_token_cbga_assp_20260810_164952`
- Checkpoint：`/home/bip/cry/code/TIRST-SAM_TMM_code/outputs_model1_gpt5p6_token_cbga_assp_formal`

### 3.1 NUDT-SIRST 失效分析

修改模型前已审计 GPT/CLIP token 缓存：1,327 个样本及其张量均为有限值，attention mask 长度有效，全局特征和 token 特征均会随样本变化。因此，特征缓存不是第 11 轮发生坍塌的原因。

根因位于门控 CBGA 的残差路径。旧实现虽然对 cross-attention delta 使用了 gate，但输出投影接收的是完整更新后的 hidden state。即使跨模态 gate 几乎关闭，模块仍然可以学习较大的单模态残差，并在 12 个 ViT blocks 中反复注入。除此之外，Linear 层的 bias 也可以绕过 gate 注入固定残差。

稳定版 `--bifusion_gate_delta_only` 只投影经过 gate 的跨模态 delta，并在稳定路径中排除输出投影 bias。因此，当 gate 关闭时，该路径严格等价于恒等映射。默认行为仍保留旧版路径，以保证历史 checkpoint 可以按照原架构加载。

### 3.2 稳定性与显存验证

| 检查项目 | 配置 | 核验结果 | 状态 |
| --- | --- | --- | --- |
| 坍塌点检查 | batch 4，冻结编码器，15 epochs；delta-only 初步修复 | 第 15 轮：mIoU 69.50、nIoU 72.76、F1 82.21、Pd 98.41、Fa 59；第 11 轮以后指标仍持续变化 | 已完成，未坍塌 |
| 编码器解冻与显存检查 | batch 2，梯度累积 2，梯度裁剪 1.0，第 2 轮解冻编码器 | 第 3 轮：mIoU 58.76、nIoU 63.74、F1 75.41、Pd 97.46、Fa 136；显存约 10 GiB，无 OOM、NaN 或 Inf | 已通过 |
| 最终 bias-free 正式路径 | batch 2，梯度累积 2，梯度裁剪 1.0 | 当前已到第 678 轮；最佳第 606 轮：mIoU 92.29、nIoU 93.38、F1 96.22、Pd 99.68、Fa 4.87 | 运行中；已稳定通过第 60 轮编码器解冻点 |

24 GiB 显存安全的正式重跑采用 batch size 2、两步梯度累积（等效 batch size 4）、梯度裁剪 1.0、最终 bias-free delta-only CBGA，以及不变的全局 ASSP prompt。该实验从 EfficientSAM baseline 重新开始，没有复用无效的旧版 NUDT checkpoint。

正式重跑产物（截至 2026-08-12 08:48 CST）：

- 进程 PID：`615615`
- 进度：678/1000 epochs
- 当前最佳：第 606 轮，mIoU 92.29、nIoU 93.38、F1 96.22、Pd 99.68、Fa 4.87
- 稳定性：日志中未发现 OOM、非有限值或 Traceback，且已稳定通过第 60 轮编码器解冻点
- 日志：`/home/bip/cry/code/TIRST-SAM_TMM_code/job_logs/model1_gpt5p6_token_cbga_assp_stable_biasless_20260811_100736/NUDT-SIRST.log`
- Checkpoint：`/home/bip/cry/code/TIRST-SAM_TMM_code/outputs_model1_gpt5p6_token_cbga_assp_stable_biasless_formal`
- 实验名称：`NUDT-SIRST_Model1_GPT5p6_tokenCBGA_deltaOnlyBiasless_globalASSP_noGT_bs2acc2_fromBaseline_split50_50`

## 四、E1 与 E2 配对消融

`ΔmIoU/nIoU/F1/Pd` 为正表示 E2 更高；`ΔFa` 为负表示 E2 的虚警更少。IRSTD-1k 和 NUAA-SIRST 使用旧版门控 CBGA；NUDT-SIRST 使用修复后的 delta-only、bias-free CBGA，因此三者不能被表述为完全相同实现下的跨数据集结果。NUDT-SIRST 仍是临时结果。

| 数据集 | ΔmIoU | ΔnIoU | ΔF1 | ΔPd | ΔFa | 解释 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| IRSTD-1k | -0.92 | -1.17 | -2.13 | -2.51 | -12.45 | 已完成：虚警明显减少，但 IoU、F1 和 Pd 均降低 |
| NUAA-SIRST | +2.33 | +1.40 | -1.50 | -0.55 | -15.55 | CBGA 提升 IoU 并显著抑制虚警，但 F1 和 Pd 略有下降 |
| NUDT-SIRST | -0.79 | -0.52 | -0.28 | +0.31 | +2.66 | 稳定版临时结果：与 E1 接近，Pd 略高，但 IoU/F1 略低且 Fa 更高 |

当前证据不支持“token-level CBGA 在所有数据集上均有提升”这一结论。IRSTD-1k 的完整结果显示 CBGA 减少了虚警，但其他主要指标下降；NUAA-SIRST 显示 mIoU、nIoU 和 Fa 改善，但 F1 和 Pd 略降；NUDT-SIRST 稳定版当前与 E1 接近，但尚未训练完成。由于 IRSTD/NUAA 与 NUDT 使用的 CBGA 残差路径不同，在完成架构对齐重跑前不能把差异仅归因于数据集。

## 五、此前完成的纯图像 TASSG 结果

这些实验只在训练阶段使用缓存的 Qwen-CLIP 特征作为蒸馏目标。部署推理采用 `semantic_source=student`，只需要红外图像，不需要文本输入。由于它们未使用新的 GPT 教师，且不属于架构对齐的 GPT 与 Qwen 配对比较，因此单独记录。

| 方法 | 数据集 | 最佳 epoch | mIoU | nIoU | F1 | Pd | Fa |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 单阶段 TASSG | IRSTD-1k | 664 | 73.89 | 75.20 | 56.26 | 94.41 | 36.06 |
| 单阶段 TASSG | NUAA-SIRST | 863 | 75.64 | 78.07 | 81.61 | 94.80 | 24.03 |
| 单阶段 TASSG | NUDT-SIRST | 866 | 92.41 | 93.14 | 96.08 | 99.47 | 2.30 |
| 两阶段 TASSG + CBGA | IRSTD-1k | 637 | 73.57 | 75.20 | 57.50 | 96.28 | 21.64 |
| 两阶段 TASSG + CBGA | NUAA-SIRST | 1119 | 75.18 | 77.97 | 83.57 | 98.04 | 30.80 |
| 两阶段 TASSG + CBGA | NUDT-SIRST | 739 | 92.17 | 93.09 | 96.15 | 99.79 | 2.53 |

## 六、证据状态与后续实验

1. 等待稳定版 E2 NUDT-SIRST 正式重跑完成 1000 epochs，再用最终最佳 checkpoint 固化结果。
2. 使用相同的 delta-only、bias-free CBGA 在 IRSTD-1k 和 NUAA-SIRST 上进行架构对齐重跑；否则不能将三数据集结果合并为同一项消融。
3. 运行架构完全匹配的 Qwen 全局特征 → ASSP 教师基线。缺少该配对基线时，不能把 GPT 与 Qwen 的性能差异直接归因于语言模型质量。
4. 使用 GPT 教师重新训练纯图像 TASSG student，并与上述已完成的 Qwen 蒸馏结果比较。
5. 补充定性可视化、困难样本和失败案例，解释 Fa 与 IoU/Pd 之间的权衡。
6. 最终论文结论至少报告三个随机种子，或提供置信区间。

原论文中的完整 CBGA+ASSP 结果使用了不同的提示和训练路径，因此没有混入上述配对表格。只有在评测协议对齐完成后，才能将其作为历史结果进行补充展示。
