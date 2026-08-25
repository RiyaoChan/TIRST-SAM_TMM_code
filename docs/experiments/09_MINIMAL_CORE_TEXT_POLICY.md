# GPT 文本最小核心属性自动筛选方案（v3）

## 1. 修订原因

旧方案把“GPT 整句是否完全正确”和“哪些文本字段足以给模型提供有效语义”混为一体，导致以下问题：

- GT 与 GPT 都描述 2 个目标时，count 已经可以由 GT 自动确认，却仍因 shape、contrast 等字段进入人工队列；
- 多目标图像通常无法用一个 shape 或 contrast 同时准确描述每个目标；
- shape、background、contrast 的人工判断成本高、主观性强，也不是给 SAM 产生粗 prompt 的必要前提；
- 用 GT 人工改写 GPT 错误字段会把 GPT 条件 `C` 混成 GT-derived oracle 条件 `O`。

因此 v3 改为“最小充分核心属性 + 字段 mask”，不再追求整句文本的每个字段都正确。

## 2. v3 自动策略

| 情形 | 自动状态 | C_core 处理 | Mandatory 人工审核 |
|---|---|---|---:|
| GT presence 与 GPT presence 不一致 | `reject_auto` | 样本不进入 C_core | 0 |
| presence 一致、GT count 与 GPT count 一致 | `presence_count_verified_auto` | 保留 presence/count | 0 |
| presence 一致、count 不一致 | `presence_only_auto` | 屏蔽 count，只保留 presence | 0 |
| location 自动一致且不靠近 3×3 边界 | `field_pass` | 可作为附加字段 | 0 |
| location 冲突或边界不确定 | `field_masked` | 屏蔽，不用 GT 改写 | 0 |
| size 自动一致 | `field_pass` | 可作为附加字段 | 0 |
| size 冲突 | `field_masked` | 屏蔽，不用 GT 改写 | 0 |
| shape/background/contrast | `excluded_by_policy` | 首轮全部屏蔽 | 0 |

样本级可用性的首要依据是 presence；presence/count 均一致时定义为核心自动通过。location/size 只决定附加字段是否保留，不再决定是否送人工审核。

## 3. 多目标规则

当 GT count 和 GPT count 都为 2 时，count 字段自动判定正确。多目标样本采用以下规则：

- 不要求人工分别判断每个目标的 shape；
- 不要求使用一个 contrast 标签概括所有目标；
- `shape/background/contrast` 的 mask 固定为 0；
- location 只有在 GPT 与 GT 的图像级粗位置一致且不处于边界时才保留；
- size 只有在 GPT 与 GT 的聚合类别一致时才保留；不一致直接屏蔽。

这意味着“两个目标数量正确”不会因为无法描述两个目标的形状或对比度而进入人工审核。

## 4. 当前自动筛选结果

总计 1,676 张训练图像：

| 数据集 | Presence+Count 自动通过 | 仅 Presence 可用 | Presence 冲突自动拒绝 | 总数 |
|---|---:|---:|---:|---:|
| IRSTD-1k | 630 | 131 | 39 | 800 |
| NUAA-SIRST | 192 | 14 | 7 | 213 |
| NUDT-SIRST | 502 | 96 | 65 | 663 |
| 合计 | 1,324 | 241 | 111 | 1,676 |

- 可用 `C_core`：1,565 张；
- Mandatory 人工审核：0 张；
- 多目标样本：577 张；
- location 自动保留：708 张；
- size 自动保留：699 张；
- shape/background/contrast 自动保留：0 张。

以上均为从现有自动审计记录计算的真实数量，不包含人工估计。

## 5. C_core 内容

`C_core` 只由 GPT 原始字段组成，GT 只控制 mask，不提供替换值。例如：

```text
infrared small target is visible; count=2; location=multiple-regions; size=tiny
```

若 count 冲突：

```text
infrared small target is visible
```

禁止把 GT count、GT location 或 GT size 写回 GPT 文本，否则应单独标记为 oracle 条件 `O`。

## 6. 对反事实蒸馏的影响

首轮反事实条件应围绕可靠核心字段构造：

- `C`：自动筛选后的 `C_core`；
- `N`：无文本或零文本特征；
- `S`：在有效 `C_core` 间随机打乱；
- `W`：只反转或扰动已验证的 presence/count，以及被 mask 保留的 location/size；
- `O`：GT 渲染的完整 oracle 文本，必须单独标记，不进入可部署主结果。

首轮不构造 shape/background/contrast 反事实。待验证 `C_core` 相对 `N` 确有稳定增益后，再通过消融实验决定是否恢复这些辅助字段。
