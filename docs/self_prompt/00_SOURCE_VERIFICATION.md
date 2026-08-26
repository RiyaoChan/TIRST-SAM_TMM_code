# Self-Prompt 文献身份、全文与代码核验

核验日期：2026-08-26。本文只把作者主页、出版社、会议论文集、arXiv、PubMed/PMC 与作者仓库作为身份和全文证据；搜索聚合页只用于发现线索。任务书共列出 68 个编号（S01–S18、A01–A50），去重后为 **53 篇独立工作**。其中 S 级 18 篇已全部核验身份，14 篇取得并阅读全文。

证据标签：`全文`=正文可逐节阅读；`摘要+代码`=只能核对摘要与真实代码，不据此填写论文消融；`摘要`=只有元数据/摘要；`排除`=来源属于本次检索策略明确排除的 MDPI，不进入证据和新颖性判断。`未发现根许可证`不等于没有版权，而是不能推定代码可复制。

## 一、必须由用户补齐或确认的材料

| ID | 材料 | 当前状态 | 请用户提供什么 |
|---|---|---|---|
| S12 | PromptPilot: Game-Theoretic Multi-Agent Prompt Optimization for Segment Anything | OpenReview PDF 被人机验证拦截；作者代码与 README 可访问 | 论文 PDF、supplementary（如有） |
| S14 | Semantic AutoSAM: Self-Prompting Segment Anything Model for Semantic Segmentation of Medical Images | IEEE EMBC 2024，PubMed 仅摘要，未找到作者公开全文/代码 | 4 页正式 PDF |
| S15 | A Unified SAM-Guided Self-Prompt Learning Framework for Infrared Small Target Detection（SAM-SPL） | IEEE TGRS 正文闭源；作者代码可访问 | 正式 PDF 与 supplementary（如有） |
| Code-08 | AlignSAM 官方仓库 `Duojun-Huang/AlignSAM-CVPR2024` | GitHub 返回 404；CVPR 正文与 supplement 已读 | 若作者曾发布代码，请提供源码压缩包或新仓库链接 |

S16 IR-SAM2、A01 PMG-SAM、A02 LDFSAM 属于 MDPI 来源，按本轮文献检索策略不纳入阅读、引用、评分和新颖性结论。若确实要比较，请另行明确要求，我会把它们放入**用户指定的独立附录**，不与主证据池混合。

### 可选补齐的 A 级正式全文

这些不是本轮S级交付的阻塞项，但目前只能依据正式摘要/元数据做机制筛查。若你方便下载，后续可以把它们加入扩展精读：

| ID | 正式题名 | DOI/入口 | 建议文件 |
|---|---|---|---|
| A05 | MUP-SAM: Multi-scale Vision Mamba UNet Prompt Generation for SAM in Multi-organ Medical Image Segmentation | DOI `10.1016/j.neunet.2026.109106` | 正式PDF+supplement |
| A08 | Taming Large Vision Model for Medical Image Segmentation via Dual Visual Prompt Tuning | DOI `10.1016/j.compmedimag.2025.102608` | 正式PDF+supplement |
| A19 | AutoPromptSeg: Automated Decoupling of Uncertainty Prompts with SAM for Semi-supervised Medical Image Segmentation | DOI `10.1016/j.compmedimag.2026.102708` | 正式PDF+supplement |
| A43 | SAM-RSIS: Progressively Adapting SAM With Box Prompting to Remote Sensing Image Instance Segmentation | DOI `10.1109/TGRS.2024.3460085` | 正式PDF+supplement |
| A50 | S4M: 4-Points to Segment Anything | DOI `10.1007/s11548-026-03689-x` | 正式PDF+supplement |

建议优先级仍是：S15 SAM-SPL > S12 PromptPilot > S14 Semantic AutoSAM > 上述A级全文。SAM-SPL与当前IR纯视觉self-prompt冲突最直接。

## 二、S 级身份与全文核验

| ID | 核验后的正式题名 | 作者 | 年份/状态 | 标识与主来源 | 全文证据 | 代码/许可 |
|---|---|---|---|---|---|---|
| S01 | SPARK-SAM: Self-Prompt Adaptation with Response Knowledge for SAM in Infrared Small Target Segmentation | Aji Mao; Zhenming Peng; Bailin Mu; Tian Pu | 2026，preprint | [arXiv:2608.20754](https://arxiv.org/abs/2608.20754) | 全文，9 页 | [SPARK-SAM](https://github.com/Sakauma/SPARK-SAM)，`adceab6`，2026-07-27；未发现根许可证 |
| S02 | IP-SAM: Rethinking Prompt-Conditioned Segmentation for Prompt-Absent Deployment | Huiyao Zhang; Jin Bai; Rui Guo; Jianwen Tan; Hongfei Wang; Ye Li | 2026，preprint | [arXiv:2603.27250](https://arxiv.org/abs/2603.27250) | 全文，含 supplement 共 28 页 | 未核验到任务书指定官方代码 |
| S03 | MaskSAM: Towards Auto-prompt SAM with Mask Classification for Volumetric Medical Image Segmentation | Bin Xie; Hao Tang; Bin Duan; Dawen Cai; Yan Yan; Gady Agam | ICCV 2025，peer-reviewed | [arXiv:2403.14103](https://arxiv.org/abs/2403.14103) | 全文 | [MaskSAM](https://github.com/bxie9/MaskSAM)，`c6860c5`，2025-10-22；未发现根许可证 |
| S04 | Sam2Rad: A Segmentation Model for Medical Images with Learnable Prompts | Assefa Seyoum Wahd et al. | CIBM 2025，peer-reviewed | [DOI:10.1016/j.compbiomed.2025.109725](https://doi.org/10.1016/j.compbiomed.2025.109725)，[arXiv:2409.06821](https://arxiv.org/abs/2409.06821) | 全文，23 页 | [SamRadiology](https://github.com/aswahd/SamRadiology)，`ccf881d`，2024-11-11；未发现根许可证 |
| S05 | AoP-SAM: Automation of Prompts for Efficient Segmentation | Yi Chen; Muyoung Son; Chuanbo Hua; Joo-Young Kim | AAAI 2025，peer-reviewed | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/32228) | 全文，9 页 | [AoP-SAM](https://github.com/Yi1-Chen/AoP-SAM)，`de503b2`，2025-10-01；Apache-2.0 |
| S06 | RSPrompter: Learning to Prompt for Remote Sensing Instance Segmentation based on Visual Foundation Model | Keyan Chen et al. | TGRS 2024，peer-reviewed | [arXiv:2306.16269](https://arxiv.org/abs/2306.16269) | 全文，含 supplement | [RSPrompter](https://github.com/KyanChen/RSPrompter)，`7c676fe`，2024-06-29；Apache-2.0 |
| S07 | UN-SAM: Universal Prompt-Free Segmentation for Generalized Nuclei Images | Zhen Chen; Qing Xu; Xinyu Liu; Yixuan Yuan | MedIA 2025，peer-reviewed | [arXiv:2402.16663](https://arxiv.org/abs/2402.16663) | 全文 | [UN-SAM](https://github.com/CUHK-AIM-Group/UN-SAM)，`1c16218`，2025-06-25；MIT |
| S08 | De-LightSAM: Modality-Decoupled Lightweight SAM for Generalizable Medical Segmentation | Qing Xu et al. | 2024，preprint | [arXiv:2407.14153](https://arxiv.org/abs/2407.14153) | 全文 | 任务书的 `ESP-MedSAM` 仓库现指向 [De-LightSAM](https://github.com/xq141839/De-LightSAM)，`d2260d7`，2025-11-07；Apache-2.0 |
| S09 | AutoPrompt-SAM3D: Integrated Generation and Selection for SAM2-Based 3D Medical Segmentation | Wanqiu Cheng; Jintao Tang; Ting Wang; Shasha Li; Ting Deng | BMC Bioinformatics 2026，peer-reviewed | [DOI:10.1186/s12859-026-06390-7](https://doi.org/10.1186/s12859-026-06390-7)，PMID 41904386 | PMC 全文 HTML（PDF 端有人机验证） | 未发现任务书指定官方仓库；文章 CC BY-NC-ND 4.0 |
| S10 | EviPrompt: A Training-Free Evidential Prompt Generation Method for Segment Anything Model in Medical Images | Yinsong Xu; Jiaqi Tang; Aidong Men; Qingchao Chen | TIP 2024，peer-reviewed | [arXiv:2311.06400](https://arxiv.org/abs/2311.06400) | 全文 | [EviPrompt](https://github.com/SPIresearch/EviPrompt)，`b8a25a5`，2024-11-13；未发现根许可证 |
| S11 | AlignSAM: Aligning Segment Anything Model to Open Context via Reinforcement Learning | Duojun Huang et al. | CVPR 2024，peer-reviewed | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Huang_AlignSAM_Aligning_Segment_Anything_Model_to_Open_Context_via_Reinforcement_CVPR_2024_paper.html) | 正文+supplement | 指定仓库 404；不可审计 |
| S12 | PromptPilot: Game-Theoretic Multi-Agent Prompt Optimization for Segment Anything | Guangze Shi et al. | ICML 2026，accepted | [OpenReview](https://openreview.net/forum?id=H6T8ECJafn) | 摘要+代码；PDF 被验证页拦截 | [PromptPilot](https://github.com/L-AILab/PromptPilot)，`c99739e`，2026-07-13；未发现根许可证 |
| S13 | Unleashing the Potential of SAM for Medical Adaptation via Hierarchical Decoding（H-SAM） | Zhiheng Cheng et al. | CVPR 2024，peer-reviewed | [arXiv:2403.18271](https://arxiv.org/abs/2403.18271) | 正文+supplement | [H-SAM](https://github.com/Cccccczh404/H-SAM)，`5bdf491`，2024-06-25；MIT |
| S14 | Semantic AutoSAM: Self-Prompting Segment Anything Model for Semantic Segmentation of Medical Images | Assefa S. Wahd; Jessica Kupper; Jacob L. Jaremko; Abhilash R. Hareendranathan | EMBC 2024，peer-reviewed | [DOI:10.1109/EMBC53108.2024.10782494](https://doi.org/10.1109/EMBC53108.2024.10782494)，PMID 40040072 | 摘要；全文待用户提供 | 未找到官方代码 |
| S15 | A Unified SAM-Guided Self-Prompt Learning Framework for Infrared Small Target Detection | Yimin Fu; Jialin Lyu; Peiyuan Ma; Zhunga Liu; Michael K. Ng | TGRS 2025，peer-reviewed | [DOI:10.1109/TGRS.2025.3610919](https://doi.org/10.1109/TGRS.2025.3610919) | 摘要+代码；正文待用户提供 | [SAM-SPL](https://github.com/fuyimin96/SAM-SPL)，`1bde7b5`，2025-12-19；未发现根许可证 |
| S16 | IR-SAM2: Target Enhancement with SAM2 for Infrared Small Target Detection | 元数据已定位 | Remote Sensing 2026，peer-reviewed | 任务书链接 | 排除：MDPI | 不进入本轮代码/证据判断 |
| S17 | Temporal-Emerged Prompting for Segment Anything in Multiframe Infrared Small Target Detection | Yinghui Xing; Donghao Chu; Shizhou Zhang; Di Xu | ICML 2026，accepted | [arXiv:2606.27655](https://arxiv.org/abs/2606.27655) | 全文+supplement | 论文声明代码公开；本轮指定 15 仓库不含该项，未做代码审计 |
| S18 | SAM-DAQ: Segment Anything Model with Depth-Guided Adaptive Queries for RGB-D Video Salient Object Detection | Jia Lin et al. | AAAI 2026，accepted | [arXiv:2511.09870](https://arxiv.org/abs/2511.09870) | 全文 | 未核验到任务书指定官方代码 |

### 任务书中的题名/出处修正

- S02 不是任务书写的 “Prompt-Space Conditioning for Prompt-Absent Camouflaged Object Detection”，arXiv 正式题名为 *Rethinking Prompt-Conditioned Segmentation for Prompt-Absent Deployment*。
- S03 的正式题名包含 “Towards” 和 “Volumetric Medical Image Segmentation”。
- S07 的正式题名是 *Universal Prompt-Free Segmentation for Generalized Nuclei Images*，不是 “Domain-Adaptive Self-Prompt ...”。
- S08/A06/A11 对应的是 **De-LightSAM**；arXiv 与当前仓库都不是 “ESP-MedSAM”。
- S15 的 venue 为 IEEE TGRS 2025；S15 正文不可开放获取，不能把 README 数字当成已精读论文结果。
- A05 MUP-SAM 发表在 *Neural Networks* 2026（DOI 10.1016/j.neunet.2026.109106），不是任务书写的 CIBM。

## 三、A 级条目身份筛查

下表保留任务书编号；“同 Sxx”表示同一工作，不重复计数。此层用于机制边界筛查，不把摘要当全文。

| ID | 核验结果/稳定入口 | 状态 | 证据边界 |
|---|---|---|---|
| A01 | PMG-SAM，Sensors 2026 | 排除 | MDPI，不进入主证据池 |
| A02 | LDFSAM，Journal of Imaging 2026，PMID 41745439 | 排除 | MDPI，不进入主证据池 |
| A03 | Self-Prompt SAM: Medical Image Segmentation via Automatic Prompt SAM Adaptation，[arXiv:2502.00630](https://arxiv.org/abs/2502.00630) | preprint | 正式题名已核验 |
| A04 | Diffusion-empowered AutoPrompt MedSAM，[arXiv:2502.06817](https://arxiv.org/abs/2502.06817) | preprint | 代码存在但公开实现不完整 |
| A05 | MUP-SAM，*Neural Networks* 2026，PMID 42269193 | peer-reviewed | 任务书 venue 修正 |
| A06 | 同 S08 | preprint | 去重 |
| A07 | PA-SAM，[arXiv:2401.13051](https://arxiv.org/abs/2401.13051) | preprint | 官方仓库可访问 |
| A08 | DVPT: Taming Large Vision Model for Medical Image Segmentation via Dual Visual Prompt Tuning，PMID 40695060 | peer-reviewed | 题名补全 |
| A09 | Hierarchical Self-Prompting SAM: A Prompt-Free Medical Image Segmentation Framework，[arXiv:2506.02854](https://arxiv.org/abs/2506.02854) | preprint | 题名修正 |
| A10 | SurgicalSAM，[AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/28514) | peer-reviewed | 代码 MIT，已审计 |
| A11 | 同 S08 | preprint | 去重 |
| A12 | 同 S07 | peer-reviewed | 去重 |
| A13 | 同 S18 | accepted | 去重 |
| A14 | 同 S16 | 排除 | MDPI |
| A15 | 同 S03 | peer-reviewed | 去重 |
| A16 | 同 S04 | peer-reviewed | 去重 |
| A17 | Enhancing the Reliability of Segment Anything Model for Auto-Prompting Medical Image Segmentation with Uncertainty Rectification（UR-SAM），[arXiv:2311.10529](https://arxiv.org/abs/2311.10529) | preprint | 正式题名已核验 |
| A18 | FNPC-SAM，SPIE 2024，PMID 38894708/PMC11182739 | peer-reviewed/OA | 全文入口存在 |
| A19 | AutoPromptSeg，CMIG 2026，PMID 41570496 | peer-reviewed | 元数据核验 |
| A20 | UncertainSAM: Fast and Efficient Uncertainty Quantification of the Segment Anything Model，ICML 2025，[arXiv:2505.05049](https://arxiv.org/abs/2505.05049) | accepted | 后验 UQ，不是 prompt generator |
| A21 | 同 S09 | peer-reviewed | 去重 |
| A22 | 同 S10 | peer-reviewed | 去重 |
| A23 | PPD: Point Prompt Defender，[作者仓库](https://github.com/L-AILab/PPD) | CVPR 2026 代码声明 | 正文未列稳定论文入口；按代码证据使用 |
| A24 | PP-SAM: Perturbed Prompts for Robust Adaptation of SAM，[CVPRW 2024](https://openaccess.thecvf.com/content/CVPR2024W/DEF-AI-MIA/html/Rahman_PP-SAM_Perturbed_Prompts_for_Robust_Adaption_of_Segment_Anything_Model_CVPRW_2024_paper.html) | peer-reviewed workshop | 正文可读 |
| A25 | Self-Prompting Large Vision Models for Few-Shot Medical Image Segmentation，[arXiv:2308.07624](https://arxiv.org/abs/2308.07624) | preprint | 正文/代码入口存在 |
| A26 | PerSAM: Personalize Segment Anything Model with One Shot，[arXiv:2305.03048](https://arxiv.org/abs/2305.03048) | peer-reviewed | 题名修正 |
| A27 | Med-PerSAM: One-Shot Visual Prompt Tuning for Personalized Segment Anything Model in Medical Domain，[arXiv:2411.16123](https://arxiv.org/abs/2411.16123) | preprint | 正式题名已核验 |
| A28 | One Polyp Identifies All: One-Shot Polyp Segmentation with SAM via Prompt Evolution（OP-SAM），[ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Mao_One_Polyp_Identifies_All_One-Shot_Polyp_Segmentation_with_SAM_via_ICCV_2025_paper.html) | peer-reviewed | 正文可读 |
| A29 | Segment Any Tissue: One-shot Reference Guided Training-free Automatic Point Prompting for Medical Image Segmentation，DOI 10.1016/j.media.2025.103550 | MedIA 2025，peer-reviewed | 正式题名与 DOI 已核验 |
| A30 | Feature-prompting GBMSeg: One-Shot Reference Guided Training-Free Prompt Engineering for Glomerular Basement Membrane Segmentation，[arXiv:2406.16271](https://arxiv.org/abs/2406.16271) | MICCAI 2024 | 正式题名/代码入口存在 |
| A31 | PGP-SAM，[作者仓库](https://github.com/PRIS-CV/PGP-SAM) | ISBI 2025 | 代码/题名入口核验 |
| A32 | GF-SAM: Bridge the Points，[作者仓库](https://github.com/ANDYZAQ/GF-SAM) | NeurIPS 2024 | 代码/题名入口核验 |
| A33 | Memory-SAM: Human-Prompt-Free Tongue Segmentation via Retrieval-to-Prompt，[arXiv:2510.15849](https://arxiv.org/abs/2510.15849) | preprint | 题名、DINOv3+FAISS retrieval 已核验 |
| A34 | ViRefSAM: Visual Reference-Guided Segment Anything Model for Remote Sensing Segmentation，[arXiv:2507.02294](https://arxiv.org/abs/2507.02294) | preprint | 正式题名已核验 |
| A35 | 同 S11 | peer-reviewed | 去重 |
| A36 | 同 S12 | accepted | 去重；全文待补 |
| A37 | 同 S05 | peer-reviewed | 去重 |
| A38 | 同 A28 | peer-reviewed | 去重 |
| A39 | ReSAM: Refine, Requery, and Reinforce: Self-Prompting Point-Supervised Segmentation for Remote Sensing Images，[arXiv:2511.21606](https://arxiv.org/abs/2511.21606) | preprint | 正式题名已核验 |
| A40 | SAM-Aware Graph Prompt Reasoning Network for Cross-Domain Few-Shot Segmentation，DOI 10.1609/aaai.v39i6.32695 | AAAI 2025，peer-reviewed | 正式题名与官方代码 `CVL-hub/GPRN` 已定位 |
| A41 | 同 A23 | 代码声明 | 去重 |
| A42 | SAM-REF: Introducing Image-Prompt Synergy during Interaction for Detail Enhancement in SAM，[CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Yu_SAM-REF_Introducing_Image-Prompt_Synergy_during_Interaction_for_Detail_Enhancement_in_CVPR_2025_paper.html) | peer-reviewed | 正文可读 |
| A43 | SAM-RSIS: Progressively Adapting SAM With Box Prompting to Remote Sensing Image Instance Segmentation，DOI 10.1109/TGRS.2024.3460085 | TGRS 2024，peer-reviewed | 正式题名/DOI 已核验 |
| A44 | GeoSAM: Fine-tuning SAM with Multi-Modal Prompts for Mobility Infrastructure Segmentation，[arXiv:2311.11319](https://arxiv.org/abs/2311.11319) | ECAI 2025，accepted | 正式题名已核验 |
| A45 | Endow SAM with Keen Eyes: Temporal-spatial Prompt Learning for Video Camouflaged Object Detection（TSP-SAM），[CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Hui_Endow_SAM_with_Keen_Eyes_Temporal-spatial_Prompt_Learning_for_Video_CVPR_2024_paper.html) | peer-reviewed | 正式题名已核验 |
| A46 | 同 S02 | preprint | 去重 |
| A47 | SPT: Self-Perception Tuning，[AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/33420) | peer-reviewed | 正文入口核验 |
| A48 | Dual-level Adapter Boosting Prompt-free Curvilinear Structure Segmentation（SACM），CVPR 2026 | peer-reviewed | 正式 CVF 入口核验 |
| A49 | OFL-SAM2，AAAI 2026 | peer-reviewed | 官方 AAAI 入口核验 |
| A50 | S4M: 4-Points to Segment Anything，Int J CARS 2026，PMID 42144534 | peer-reviewed | 元数据核验 |

## 四、指定仓库快照与许可结论

| 仓库 | 快照 | 根许可证 | 是否可复制实现 |
|---|---:|---|---|
| SPARK-SAM | `adceab6` | 未发现 | 否；只可读审计/自行重写 |
| MaskSAM | `c6860c5` | 未发现 | 否；只可读审计/自行重写 |
| SamRadiology | `ccf881d` | 未发现 | 否；只可读审计/自行重写 |
| AoP-SAM | `de503b2` | Apache-2.0 | 可在满足许可证与 NOTICE 条件下复用 |
| RSPrompter | `7c676fe` | Apache-2.0 | 可在满足许可证条件下复用 |
| UN-SAM | `1c16218` | MIT | 可在保留许可声明下复用 |
| EviPrompt | `b8a25a5` | 未发现 | 否；只可读审计/自行重写 |
| AlignSAM | 不可访问 | 不可核验 | 否 |
| PromptPilot | `c99739e` | 未发现 | 否；只可读审计/自行重写 |
| H-SAM | `5bdf491` | MIT | 可在保留许可声明下复用 |
| SAM-SPL | `1bde7b5` | 未发现 | 否；只可读审计/自行重写 |
| AutoPromptMedSAM | `ece3ce2` | 未发现 | 否；且当前公开主类存在未连接成员，不能直接运行 |
| De-LightSAM | `d2260d7` | Apache-2.0 | 可在满足许可证条件下复用 |
| SurgicalSAM | `4b4c655` | MIT | 可在保留许可声明下复用 |
| PPD | `f040aab` | 未发现 | 否；只可读审计/自行重写 |

## 五、来源质量评分

评分为“与本任务洞察相关性/证据完整性/数值证据可核验性”，1–5 分；不是论文质量排名。

| 证据组 | 类型 | 洞察 | 完整 | 数值 | 用法 |
|---|---|---:|---:|---:|---|
| S01/S02/S03/S04/S05/S06/S07/S08/S09/S10/S11/S13/S17/S18 | 方法论文+全文 | 5 | 5 | 4–5 | 深读与机制/冲突证据 |
| S12 | 方法论文+代码 | 4 | 2 | 2 | 只陈述代码可核验事实，待 PDF |
| S14 | 方法论文摘要 | 3 | 1 | 1 | 只作身份和大类定位 |
| S15 | 方法论文摘要+代码 | 5 | 3 | 2 | 代码路径与部署事实；论文消融待 PDF |
| S16/A01/A02 | MDPI | — | — | — | 策略排除 |
| A03–A50（去重后） | 邻近方法/综述筛查 | 2–4 | 2–5 | 2–5 | 新颖性边界，不替代 S 级深读 |

## 六、可复核性说明

- 全文 PDF/HTML、浅克隆仓库和搜索缓存只保留在本机临时目录，没有加入 Git。
- 仓库路径、类名与张量 shape 见 `03_REFERENCE_CODE_PATH_AUDIT.md`；论文机制见 `02_S_TIER_DEEP_READING.md`。
- 对 S12/S14/S15，后续拿到 PDF 后应补写消融、失败案例、算力和论文—代码一致性；在补齐前不得引用它们的 README 数值作为论文证据。
