# TIRST-SAM 中文实验记录

最后更新：2026-08-25 16:00 CST

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

## 七、E3：GPT 结构化分角色 token → 双稀疏提示，不使用 GT 点提示

### 7.1 修改状态与表示方式

截至 2026-08-19，模型输入已经从“把整段固定描述压成单个 CLIP global embedding”扩展为字段可独立屏蔽的 role-token 缓存。新增生成器为 `scripts/build_structured_role_token_features.py`，首轮主实验只启用固定顺序 `[presence, count]`：

- 每个字段分别构造成自然语言短句并独立通过 CLIP ViT-B/32 文本编码器；
- 每个字段取各自归一化的 EOT/global 向量，形成一个 `[512]` role token，而不是使用整句的 77 个 CLIP wordpiece token；
- 缓存张量为 `token_features=[2,512]`、`attention_mask=[2]`，同时保存角色名、角色文本、字段值、字段 mask 和 mask 策略；
- 被自动审核判错的字段同时执行 `attention_mask=0` 和 token 向量清零，因此不会污染其他正确字段；
- 缓存仍保存 masked-mean `global_feat` 用于旧接口兼容，但本实验明确使用 `text_sparse_prompt_source=fused_tokens`，不走 `raw_global`；
- `text_sparse_num_tokens=2`，presence 和 count 分别产生一个 SAM 稀疏提示 token。

`location/size` 已由生成器支持，但未进入首轮主实验。原因是它们的自动核验稳定性低于 presence/count，先作为独立消融，不能重新混入首轮核心条件。`shape/background/contrast` 未进入该缓存。

### 7.2 训练/测试字段 mask 与无 GT 约束

训练集只用 GT mask 判断 GPT 字段是否可信，不用 GT 值替换 GPT 值：

- presence 冲突：两个 role token 都屏蔽；图像和分割 mask 仍用于分割训练，该样本等价于无文本条件；
- presence 正确、count 冲突：只保留 presence token；
- presence/count 均正确：保留两个 token；
- 测试集不运行 GT 审核，直接使用 GPT 的 presence/count 原始字段。

三个测试集 `raw_gpt_inference` 数量恰好等于各自 test split 大小，说明测试 GT 没有进入字段 mask：

| 数据集 | 总样本 | 训练审核样本 | 测试 raw GPT | 训练 presence+count | 训练 presence-only | 训练全屏蔽 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| IRSTD-1k | 1001 | 800 | 201 | 630 | 131 | 39 |
| NUAA-SIRST | 427 | 213 | 214 | 192 | 14 | 7 |
| NUDT-SIRST | 1327 | 663 | 664 | 502 | 96 | 65 |

缓存 SHA-256：

| 数据集 | 文件大小（bytes） | SHA-256 |
| --- | ---: | --- |
| IRSTD-1k | 4,749,880 | `f4c29dbc06e96046a0daff5c846c33d5aa836a940d848c312e28997623c99ad0` |
| NUAA-SIRST | 2,026,354 | `fe282a71fa0b27e384800a388a7de330e519f4dc144a1681379c095e6ff2e5b6` |
| NUDT-SIRST | 6,297,510 | `990cda88e723dcb6250582bba792da18923a6ba9b5218e117e6532f8999b5245` |

### 7.3 本地 smoke test

环境：Windows 本地 RTX 5090 D 32 GB，PyTorch 2.10.0+cu130。IRSTD-1k 完整训练集和测试集运行 1 epoch，成功完成数据加载、前向、反向、阈值搜索、物理指标评估和 checkpoint 保存。加载器实际张量形状为 `[B,2,512]` 与 `[B,2]`，启动日志确认 `source=fused_tokens, tokens=2`。

| epoch | loss | mIoU | nIoU | F1 | Pd | Fa | 用途 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1.0136 | 17.97 | 40.11 | 29.60 | 75.66 | 760.13 | 仅管线验证，不作为正式性能结论 |

新增字段 mask/缓存测试 6 项；连同原 GPT 审核测试共 `13 passed`。

### 7.4 正式训练状态

本地串行队列已于 2026-08-19 16:38 CST 启动，三个数据集均计划从同一个 `weights/efficient_sam_vitt.pt` baseline 训练 1000 epochs。统一配置为 256×256、batch size 4、HQ warm-up 30 epochs、编码器冻结 60 epochs、`prompt_mode=assp_only`、`fused_tokens` 和 2 个 role sparse tokens。

- 串行监督队列已正常结束；
- IRSTD-1k 已于 2026-08-20 04:31 CST 完成 1000 epochs；
- NUAA-SIRST 已于 2026-08-20 10:24 CST 完成 1000 epochs；
- NUDT-SIRST 已于 2026-08-20 23:30 CST 完成 1000 epochs；
- 三个日志均未发现 NaN、Inf、OOM、RuntimeError 或 Traceback；
- 2026-08-25 10:15 CST 再次交叉核验三个日志、`best.pt` 元数据与命名 checkpoint，最佳 epoch 和指标均一致；
- 正式输出：`outputs/model1_gpt5p6_rolepc_sparse2_formal`；
- 日志：`job_logs/model1_gpt5p6_rolepc_sparse2_20260819`；
- 可复现启动脚本：`scripts/tmm_train_role_tokens_local.ps1`。

结果均按验证集 mIoU 最高的 checkpoint 记录；`Fa` 单位为 `×10^-6`。

| 数据集 | 进度 | 最佳 epoch | mIoU | nIoU | F1 | Pd | Fa | 状态 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| IRSTD-1k | 1000/1000 | 768 | 74.14 | 73.39 | 61.43 | 94.61 | 8.05 | 已完成 |
| NUAA-SIRST | 1000/1000 | 991 | 74.93 | 77.75 | 82.56 | 93.60 | 39.15 | 已完成 |
| NUDT-SIRST | 1000/1000 | 982 | 93.46 | 94.35 | 96.76 | 99.37 | 1.65 | 已完成 |

最佳 checkpoint 文件：

- IRSTD-1k：`best_ep768_miou74p14_niou73p39_f161p43_pd94p61_fa8p05.pt`；
- NUAA-SIRST：`best_ep991_miou74p93_niou77p75_f182p56_pd93p60_fa39p15.pt`；
- NUDT-SIRST：`best_ep982_miou93p46_niou94p35_f196p76_pd99p37_fa1p65.pt`。

### 7.5 E3 与旧 E1 的最终配对比较

`ΔmIoU/nIoU/F1/Pd` 为正表示 E3 更高；`ΔFa` 为负表示 E3 虚警更少。

| 数据集 | ΔmIoU | ΔnIoU | ΔF1 | ΔPd | ΔFa | 当前解释 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| IRSTD-1k | -0.70 | -4.14 | +5.64 | -0.47 | -13.89 | F1 与 Fa 明显改善，但 nIoU 下降 |
| NUAA-SIRST | -0.41 | +0.25 | -1.17 | -2.48 | +11.34 | IoU 基本相当，但 Pd、F1 和 Fa 变差 |
| NUDT-SIRST | +0.38 | +0.45 | +0.26 | 0.00 | -0.56 | IoU、F1 和 Fa 小幅改善，Pd 持平 |

当前证据不支持“presence/count role token 在三个数据集上稳定优于整句 global embedding”的结论。IRSTD-1k 显示更高 F1、更低 Fa，但 nIoU 下降；NUAA-SIRST 的 IoU 接近 E1，但 F1、Pd 和 Fa 变差；NUDT-SIRST 则在 IoU、F1 和 Fa 上取得小幅改善。整体效果具有明显的数据集依赖性。

### 7.6 证据边界与待补对照

E3 与旧 E1 不能直接解释为“纯 role-token 增益”：E1 使用整段描述的单个 global embedding、1 个带门控 `raw_global` token；E3 使用两个独立字段向量和 2 个 `fused_tokens` sparse prompts。正式归因至少还需补充：

1. 相同双-token提示接口下的整句/wordpiece 对照；
2. 同架构、同训练预算的 no-text 对照；
3. presence-only 与 presence+count 的字段增量消融；
4. 三个随机种子或置信区间；
5. 使用固定 `center/resize` 而非随机验证裁剪，对所有最佳 checkpoint 进行确定性重评。

当前完成的是反事实行为蒸馏所需的“字段可隔离教师输入”。完整 `C/N/S/W Prompt→Mask` 教师行为缓存及学生蒸馏尚未开始，不能把 E3 记作反事实蒸馏结果。

## 八、E4：文本反事实 `C/N/S/W/O` 确定性诊断

2026-08-25 已使用同一个 E3 best checkpoint 在 IRSTD-1k 与 NUAA-SIRST 上完成低成本诊断。五种条件共享同一次 image encoder forward；评测固定 `sc_eval_crop=resize`，使用零长度 point prompts，`C/N/S/W` 不读取评测 GT，只有 `O` 使用 GT presence/count 并标为不可部署上界。下表统一采用 `N` 条件选择的固定阈值。

| 数据集 | 条件 | global IoU | mean IoU | F1 | Pd | Fa | mask AUPRC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| IRSTD-1k | C | 61.28 | 51.71 | 63.78 | 84.88 | 22.39 | 74.36 |
| IRSTD-1k | N | 60.04 | 51.81 | 64.01 | 86.25 | 28.16 | 72.34 |
| IRSTD-1k | S | 61.26 | 51.69 | 63.77 | 84.88 | 22.39 | 74.37 |
| IRSTD-1k | W | 61.28 | 51.71 | 63.78 | 84.88 | 22.39 | 74.36 |
| IRSTD-1k | O | 61.30 | 51.72 | 63.79 | 84.88 | 22.24 | 74.36 |
| NUAA-SIRST | C | 66.51 | 62.93 | 75.76 | 92.78 | 3.57 | 78.09 |
| NUAA-SIRST | N | 66.49 | 62.91 | 75.74 | 92.78 | 3.57 | 78.08 |
| NUAA-SIRST | S | 66.51 | 62.93 | 75.76 | 92.78 | 3.57 | 78.09 |
| NUAA-SIRST | W | 66.51 | 62.93 | 75.76 | 92.78 | 3.57 | 78.09 |
| NUAA-SIRST | O | 66.51 | 62.93 | 75.76 | 92.78 | 3.57 | 78.09 |

逐图 `C-N` mean-IoU 差值：IRSTD-1k 为 -0.10 个百分点，bootstrap 95% CI `[-1.19,+0.99]`；NUAA-SIRST 为 +0.02 个百分点，95% CI `[-0.02,+0.07]`。两个 CI 均跨 0。`C/S/W/O` 的 sparse embeddings 虽有差异，但最终概率图几乎不变，说明当前 decoder 主要响应 token 是否激活，而没有可靠使用文本内容或图文对应关系。

因此暂停 Prompt→Mask 反事实增量蒸馏，不启动 B4/B5。完整协议、逐图统计、表示距离与产物路径见 `docs/experiments/10_TEXT_COUNTERFACTUAL_DIAGNOSTIC_20260825.md`。

### 8.1 匹配 no-text 筛选探针

2026-08-25 12:35 CST，IRSTD-1k 100-epoch null-token 训练探针已完成。模型保留与 E3 相同的 2-token projector 及 395,520 个参数，仅将输入 role features 与 attention mask 清零；从同一 baseline 初始化，验证固定 `sc_eval_crop=resize`。输出位于 `outputs/model1_matched_null_probe_20260825`，日志位于 `job_logs/model1_matched_null_probe_20260825`，stderr 为空，无 NaN、Inf、OOM、RuntimeError 或 Traceback。

训练日志中的最佳 checkpoint 为 epoch 95：global IoU=59.22、mean IoU=49.46、F1=61.66、Pd=83.85、Fa=23.61，阈值为 0.41。第 100 轮为 global IoU=57.79、mean IoU=49.76、F1=62.01、Pd=89.00、Fa=39.00，依照预注册规则仍保留 epoch 95。

为统一评测实现，又使用 E4 确定性评测器分别评浏 null-token epoch 95 的 `N` 条件和 E3 epoch 97 的 `C` 条件。两者均为 100-epoch 筛选范围内的最佳已保存 checkpoint，固定 `resize`，且各自选择一个数据集级固定阈值：

| 100-epoch 探针 | 条件 | 阈值 | global IoU | mean IoU | F1 | Pd | Fa | mask AUPRC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E3 role token, epoch 97 | C | 0.95 | 55.47 | 50.01 | 61.96 | 85.22 | 54.05 | 69.98 |
| matched null token, epoch 95 | N | 0.30 | 58.17 | 48.93 | 61.23 | 84.88 | 27.41 | 72.77 |

与 null-token 相比，E3-100 的 mean IoU 仅高 1.08 个百分点，但 global IoU 低 2.70 个百分点、Fa 高 26.65、AUPRC 低 2.79 个百分点，是明显的混合胜负。原 E3 的 checkpoint 是依随机验证裁剪保存，null-token 依固定 `resize` 保存，且当前只有单种子，因此该表只能用于筛选，不能声称任一方显著更优。

结合 8.0 中 `C≈S≈W≈O` 的直接证据，当前不将上述差异归因于文本语义，也不扩展 NUAA-SIRST/null-token 三种子。反事实行为蒸馏保持停止，下一主线转向高分辨率视觉 self-prompt；文本只保留为需先证明能验证候选或降低 Fa 的可拒绝可选 gate。

## 九、Experiment 1：可靠性校准的视觉 Self-Prompt

2026-08-25 已完成旧 self-prompt 路径审计、干净 split、统一 proposal/metrics、无 GT 泄漏测试、20-epoch early/mid/neck probe 和 IRSTD-1k prompt-level A1–A3。完整记录见：

- `docs/experiments/11_SELF_PROMPT_CODE_AND_PROTOCOL_AUDIT.md`；
- `docs/experiments/12_PROMPT_GENERATOR_SCREENING.md`；
- `docs/experiments/13_MULTIVIEW_RELIABILITY_EXPERIMENT.md`。

P0 在 IRSTD-1k 新 validation（80 张）选择 neck probe：tiny Recall@20=84.75%，overall Recall@20=92.31%，False Prompts/MP=220.87，Dense AUPRC=41.52%。它在 tiny recall 上与 DoG/LoG 并列，但 overall recall、False Prompts 和 Dense AUPRC 更优，因此不新增 NativeResolutionPromptHead。

五视图 A2 将 overall/tiny Recall@20 提高到 94.02%/89.83%，但 False Prompts/MP 升到 256.16。只在 validation 的缓存 cluster 上搜索 108 个规则组合后，A3 `alpha=0.5, beta=0, gamma=0, min_support=2/5, max_dispersion=2 px, tau=0` 达到 Recall@20=97.44%、tiny Recall@20=94.92%、False Prompts/MP=223.16、Candidate AUPRC=69.92%；相对 A2 虚警候选下降 12.88%，召回没有下降，且 AUPRC 高于 A2-max 的 61.15%。默认 `min_support=3/5` 虽虚警更低，但召回下降超过闸门限制，未被选择。

100-epoch mask 筛选与统一评测已于 2026-08-26 完成。所有数值来自 IRSTD-1k 新 validation（80 张）、seed 20260825、固定 resize 与 segmentation threshold 0.5；`Fa` 单位为 `×10^-6`。A2/A3 复用 A1-P 的 best-mask checkpoint，只改变推理期 prompt 形成方式。

| Run | Prompt | Best epoch | global IoU | mean nIoU | F1 | Pd | Fa | latency (ms/image) | 状态 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A0 | null/no spatial prompt | 82 | 56.02 | 47.86 | 60.34 | 88.89 | 50.54 | 26.88 | 完成 |
| A1-P | 单视图 positive points | 84 | 55.61 | 48.13 | 60.20 | 88.03 | 44.06 | 25.10 | 完成；A2/A3 主线 |
| A1-D | 单视图 dense targetness | 88 | 55.62 | 49.52 | 61.37 | 86.32 | 39.67 | 25.07 | 完成 |
| A2 | 五视图 mean/max points | 84 | **56.09** | 48.01 | 60.11 | 86.32 | **33.57** | 106.25 | 完成 |
| A3 | 五视图 rule-gated points | 84 | 55.58 | **48.70** | **60.86** | **87.18** | 40.44 | 106.52 | 完成 |

A1-P 和 A1-D 均没有整体超过 A0。A1-DP 在 60/100 epochs 后于第 61 轮发生 CUDA OOM；其部分 checkpoint 不进入正式比较。因为预注册主线要求逐点拒绝，而 A1-D 没有显著更强且未实现逐样本 `dense_prompt_valid`，A1-DP 不可能改变 A1-P 主线选择，所以不重跑。

A2 相对 A1-P 将 global IoU 提高 0.49pp、Fa 降低 23.81%，但 Pd 下降 1.71pp；相对 A0 的 IoU 只高 0.07pp，Pd 低 2.56pp。A3 相对 A2 的 prompt-level False Prompts/MP 下降 12.88%、Recall@20 提高 3.42pp，但最终 mask 的 global IoU 下降 0.52pp、Fa 增加 20.45%，只有 mean nIoU、F1 和 Pd 上升。由此只能主张多视图/规则改变了低虚警—检出权衡，不能主张 A3 全面改善最终分割。

按预注册 A3 六项闸门，IRSTD prompt-level 的五项可计算条件通过，但 NUAA-SIRST 同规则方向尚未验证；同时 mask-level 收益不稳定。因此 A3 完整闸门未通过，A4、NUDT-SIRST 与多随机种子长训练均未启动。当前证据指向 prompt encoder/decoder 对候选排序变化响应不足，A3 暂作为分析模块而不是主模型。

关键 best-mask SHA-256：A0 `722c7f7838aeca5517a54af6791662342aade8222392243a7db9111bfe39388b`，A1-P `6320c5e2a68aa934b92b869998d826463b630f560f96e4257391deebabc9a904`，A1-D `e94fefa22cd116f149f197544d90e5c9197dfa6f8ace9918d9717853a728ead8`。完整分层结论、失败边界、运行路径与复现命令见 `docs/experiments/13_MULTIVIEW_RELIABILITY_EXPERIMENT.md`。

## 十、Self-Prompt 文献深读与下一阶段机制闸门

2026-08-26 按 `experiments_guide/TIRST_SAM_SELF_PROMPT_LITERATURE_READING_FOR_CODEX.md` 完成文献身份核验、S级全文阅读、指定仓库代码审计、当前失败对照、候选Idea与Top-3最小实验设计。本节**只记录文献/机制结论，没有新增训练或性能结果，也未修改核心模型**。

交付文档：

- `docs/self_prompt/00_SOURCE_VERIFICATION.md`：任务书68个编号、53篇独立工作的身份/全文/代码/许可证核验；
- `docs/self_prompt/01_SELF_PROMPT_PAPER_MATRIX.md`：prompt表示、no-object/background、decoder消费和部署依赖矩阵；
- `docs/self_prompt/02_S_TIER_DEEP_READING.md`：18个S级条目，其中14篇基于全文；
- `docs/self_prompt/03_REFERENCE_CODE_PATH_AUDIT.md`：15个指定仓库中14个完成真实forward/loss/inference审计；
- `docs/self_prompt/04_CURRENT_FAILURE_VS_LITERATURE.md`：PR #3逐环节失败诊断；
- `docs/self_prompt/05_IDEA_CANDIDATES.md`：12个可证伪候选；
- `docs/self_prompt/06_NOVELTY_COLLISION_AUDIT.md`：与SPARK-SAM、IP-SAM、SAM-SPL、MaskSAM等的冲突边界；
- `docs/self_prompt/07_TOP3_MINIMAL_EXPERIMENTS.md`：只读/oracle优先的低成本预注册计划；
- `references/self_prompt_related.bib`：已核验的非排除来源BibTeX。

### 10.1 文献导出的主结论

1. 当前A3的核心失败不是prompt-level候选还不够准，而是候选进入SAM后丢失身份、background、no-object和reliability；`[B,1,K,2]`把多目标/错误点混入一个query。
2. “图像生成prompt”“浅层高分辨率self-prompt”“前/背景prompt”“response adaptation”“one-query-one-mask/no-object”均已有直接先例；不能把其中任何单项写成首次。
3. 下一步必须先证明decoder消费prompt：correct/zero/shuffled/wrong、reliability置零/随机、one-query vs multi-query、candidate drop和oracle micro-mask是必做反事实。
4. SAM-SPL是纯视觉IRSTD self-prompt直接基线；SPARK-SAM是IRSTD prompt–response adaptation直接近邻；IP-SAM是前/背景prompt-space直接近邻；MaskSAM/RSPrompter是object-set/no-object直接近邻。
5. 继续更换显著性算子、调Top-K/NMS、增加deterministic views或训练support/dispersion MLP均停止作为主创新。

### 10.2 Top-3待证伪方向

| 优先级 | 方向 | 一句话机制 | 首要停止条件 |
|---:|---|---|---|
| 1 | MicroQuery-SAM | 每候选独立query+micro-mask+`∅`，以objectness×reliability×SAM-IoU聚合 | oracle independent-query也不优于one-query |
| 2 | TB-Prompt | 每候选保持target/background成对状态直到decoder，并做wrong/shuffled background反事实 | correct background不优于shuffled或持续过抑制tiny目标 |
| 3 | RQ-Adapt | 用首轮独立mask response预测accept/refine/reject，而不是继续调candidate score | response AUPRC不优于原candidate score，oracle reject也不能降Fa |

### 10.3 执行闸门

下一阶段先运行零训练的M0/M1、T0、R0/R1缓存诊断。只有某条在两个validation split达到预注册Fa/Pd/IoU门槛，才进入20-epoch sanity和100-epoch三随机种子筛选。**不直接启动1000 epochs**；机制闸门不过时，延长训练没有研究价值。

### 10.4 待用户补齐材料

- PromptPilot论文PDF与supplementary（OpenReview被验证页拦截）；
- Semantic AutoSAM正式4页PDF；
- SAM-SPL正式TGRS PDF与supplementary；
- AlignSAM源码压缩包或新的官方仓库链接（指定GitHub链接404）。

IR-SAM2、PMG-SAM、LDFSAM属于本轮来源策略排除的MDPI工作，没有进入证据、引用和新颖性评分；如需比较，应单独建立用户指定附录。
