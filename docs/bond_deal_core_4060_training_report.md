# 债券成交结构抽取模型首夜训练报告

> 实验名称：`core_4060`  
> 训练时间：2026-08-26 夜间至 2026-08-27  
> 训练任务：债券成交文本 21 个核心字段的单句结构化抽取  
> 基础模型：`fastino/gliner2-base-v1`（本地快照）  
> 最终模型：`test/outputs/structure/core_4060/best`

## 1. 实验结论

本次实验完成了“领域 NER 预热 + 结构抽取微调”的两阶段全参数训练。在 1,028 条测试样本上，最终结果如下：

| 指标 | 结果 |
|---|---:|
| 单句严格完全匹配准确率 | **52.4319%** |
| 完全正确句数 | **539 / 1,028** |
| 字段级 Micro Precision | **89.2411%** |
| 字段级 Micro Recall | **89.3912%** |
| 字段级 Micro F1 | **89.3161%** |
| 最终推理阈值 | **0.45** |

这里的“单句严格完全匹配”是本项目的核心业务指标：一句中预测出的 deal 数量必须正确，并且每个 deal 的全部非空字段及字段值必须与标注完全一致。任何一个字段漏抽、误抽、边界错误或 deal 归属错误，整句均计为错误。

字段级 F1 已接近 90%，但句级准确率只有约 52%，说明模型已经具备较强的字段识别能力，当前主要瓶颈是复杂句中的字段完整性和同一成交记录内的字段绑定。

## 2. 训练环境

| 项目 | 配置 |
|---|---|
| 操作系统 | Windows（训练日志路径为 Windows） |
| Python | `D:\Anaconda\python.exe` |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| 显存 | 8.0 GiB |
| PyTorch | 2.7.1+cu126 |
| Transformers | 4.52.4 |
| PEFT | 0.20.0 |
| 混合精度 | FP16 |

## 3. 模型与本次具体修改

### 3.1 基础模型结构

本次训练使用 GLiNER2 的经典 `span` 架构，而不是实验性的 `boundary` 架构。编码器为 DeBERTa-v3-base 对应的 DeBERTa-v2 实现：

| 参数 | 值 |
|---|---:|
| 编码器隐藏维度 | 768 |
| Transformer 层数 | 12 |
| 注意力头数 | 12 |
| FFN 中间维度 | 3,072 |
| 最大位置长度 | 512 |
| 词表大小 | 128,011 |
| Hidden Dropout | 0.1 |
| Attention Dropout | 0.1 |
| Span 表示方式 | `markerV0` |
| 最大 span 宽度 | 8 个 word token |
| 计数模块 | `count_lstm_v2` |
| Token Pooling | `first` |
| Attention 实现 | SDPA |

模型先预测一句话中的结构实例数量，再根据预测数量产生实例相关的字段表示，为每个实例、每个字段在文本 span 中打分。

### 3.2 面向债券业务的模型训练改造

本次没有替换 DeBERTa 编码器，也没有启用 LoRA；真正生效的模型侧修改如下。

1. **两阶段领域迁移**

   第一阶段先使用同一批债券字段做 NER 微调，使编码器和 span head 学习债券代码、收益率、成交量、机构名称、结算方式等领域表达。第二阶段不再从原始基础模型开始，而是从第一阶段最佳 NER checkpoint 初始化，再训练 deal 结构抽取。

2. **编码器和任务头使用差分学习率**

   DeBERTa 编码器学习率为 `1e-5`，span、计数和结构任务头学习率为 `5e-4`。任务头学习率是编码器的 50 倍，使新任务头较快适应债券字段，同时减小基础语言表示被破坏的风险。

3. **保留显式字符位置监督**

   结构训练没有简单地把标注 span 转为字符串，而是保留 `{text, start, end}`。处理器直接将字符区间映射到 word span，避免一句中多次出现相同的 `1000`、`+0` 或机构名称时，因字符串搜索而绑定到错误位置。

4. **固定领域 Schema**

   训练和推理均使用相同的 21 字段中文描述。项目中的 JSON 字段随机删除、随机改名和结构 dropout 对本任务默认关闭，避免训练期 schema 与最终业务 schema 不一致。

5. **全参数微调**

   `use_lora=false`，因此不是参数高效适配器训练，而是对编码器及 GLiNER2 任务模块进行完整微调。最终 `model.safetensors` 大小约 795 MiB。

6. **面向 8 GiB 显存的训练配置**

   使用 FP16、gradient checkpointing、micro batch 2 和梯度累积 8，在控制显存的同时维持每次优化更新的有效 batch size 为 16。

需要强调：源码中已经存在更适合多 deal 无序集合匹配的 `boundary + record head`，但昨晚这次实验没有使用该架构，因此本报告结果只代表经典 `span` 路线。

## 4. 数据与本次具体修改

### 4.1 原始数据

原始文件为 `bond_deal_0805_structured_aug_v1_sample_10000.jsonl`，实际读取 10,000 条样本。原始数据包含 34,661 个 deal，单句可能包含 1～12 笔成交。

本次训练只保留以下 21 个高频核心字段：

`volume`、`bond_code`、`buyer`、`seller`、`settlement_type`、`yield`、`settlement_date`、`bond_name`、`residual_maturity`、`send_to`、`serial_number`、`send_type`、`send_to_trader`、`net_price`、`bridge_institution`、`send_from`、`source`、`rating`、`buyer_send_type`、`seller_send_type`、`bridge_trader_name`。

因此，最终 52.43% 是 **21 个核心字段指标**，不能解释为全量字段准确率。

### 4.2 数据处理改造

1. **多 deal 句拆分**

   对能够根据字段标注区间可靠切开的多成交句，拆成单 deal 子句，并重新计算所有字符偏移。若不同 deal 的标注区间重叠，则保留原始多 deal 句，防止错误裁剪。

2. **字段压缩与 Schema 对齐**

   删除核心字段集合之外的字段；训练记录仍保留核心字段中的空值，使训练 schema 的字段集合稳定。每条结构样本统一注入同一份字段中文描述。

3. **生成两种训练视图**

   - NER 数据：把每个 deal 的字段 span 展平为字段类型到实体 mention 的映射。
   - Structure 数据：保留 deal 层级、字段名和字符偏移，用于第二阶段结构绑定训练。

4. **按增强谱系进行分组切分**

   使用 fingerprint/source fingerprint 分组后按 80%/10%/10% 切分，随机种子为 42；同一 fingerprint stem 的拆分子句保持在同一集合。

5. **数据校验**

   train、validation、test 的 NER 与 structure 数据均通过格式和 span 校验，日志记录的错误数为 0。

### 4.3 处理后的数据规模

| 数据集 | 句数 | deal 数 | 多 deal 句数 | 平均字符长度 | 最大字符长度 |
|---|---:|---:|---:|---:|---:|
| Train | 8,224 | 8,453 | 93 | 55.71 | 273 |
| Validation | 1,028 | 1,044 | 8 | 55.39 | 177 |
| Test | 1,028 | 1,073 | 16 | 55.44 | 269 |
| 合计 | 10,280 | 10,570 | 117 | — | — |

训练集第一增强动作分布为：`multi_merge` 6,955 条、`split` 594 条、`tail_replace` 303 条、`noise` 289 条、`duplicate_bond_row` 83 条。

## 5. 详细训练参数

两个阶段除 epoch 数和初始化 checkpoint 外，使用相同的优化参数。

| 参数 | NER 阶段 | Structure 阶段 |
|---|---:|---:|
| Epoch | 1 | 3 |
| 训练样本数 | 8,224 | 8,224 |
| 验证样本数 | 1,028 | 1,028 |
| Micro batch size | 2 | 2 |
| Eval batch size | 4 | 4 |
| Gradient accumulation | 8 | 8 |
| 有效训练 batch size | 16 | 16 |
| 最大输入长度 | 256 | 256 |
| Encoder LR | 1e-5 | 1e-5 |
| Task head LR | 5e-4 | 5e-4 |
| Weight decay | 0.01 | 0.01 |
| Adam beta1 / beta2 | 0.9 / 0.999 | 0.9 / 0.999 |
| Adam epsilon | 1e-8 | 1e-8 |
| 最大梯度范数 | 1.0 | 1.0 |
| Scheduler | Linear | Linear |
| Warmup ratio | 0.1 | 0.1 |
| FP16 / BF16 | true / false | true / false |
| Gradient checkpointing | true | true |
| Length grouping | true | true |
| Fused optimizer | true | true |
| TF32 | true | true |
| 编译模型 | false | false |
| LoRA | false | false |
| Eval interval | 100 steps | 100 steps |
| Logging interval | 20 steps | 20 steps |
| 最佳模型指标 | 最小 `eval_loss` | 最小 `eval_loss` |
| Early stopping patience | 3 | 3 |
| 保存 checkpoint 上限 | 2 | 2 |
| 随机种子 | 42 | 42 |
| Deterministic | false | false |
| DataLoader workers | 0 | 0 |

## 6. 训练过程与收敛结果

### 6.1 第一阶段：NER 预热

| 项目 | 结果 |
|---|---:|
| 优化步数 | 514 |
| 完成 epoch | 1 |
| 训练耗时 | 977.63 秒（约 16 分 18 秒） |
| 处理速度 | 8.41 samples/s |
| 最佳 checkpoint | step 500 |
| 最佳验证损失 | 1.250106 |

NER 验证损失变化：

| Step | Eval loss |
|---:|---:|
| 100 | 5.4382 |
| 200 | 5.2164 |
| 300 | 3.0088 |
| 400 | 1.7414 |
| 500 | **1.2501** |

验证损失在整个 NER 阶段持续下降，说明一个 epoch 结束时仍未出现明显过拟合。

### 6.2 第二阶段：结构抽取

| 项目 | 结果 |
|---|---:|
| 初始化模型 | NER 阶段最佳 checkpoint |
| 优化步数 | 1,542 |
| 完成 epoch | 3 |
| 训练耗时 | 2,643.52 秒（约 44 分 04 秒） |
| 处理速度 | 9.33 samples/s |
| 最佳 checkpoint | step 1400 |
| 最佳验证损失 | 1.860278 |

关键验证点：

| Step | Eval loss |
|---:|---:|
| 100 | 12.5317 |
| 400 | 5.2568 |
| 700 | 2.9777 |
| 800 | 2.5187 |
| 1100 | 2.3444 |
| 1300 | 2.1939 |
| 1400 | **1.8603** |
| 1500 | 1.9772 |

结构阶段在 step 1400 达到最低验证损失，step 1500 出现小幅回升，因此保存的 `best` checkpoint 对应 step 1400。两个训练阶段合计耗时约 60 分 21 秒，不含数据准备和最终测试时间。

## 7. 最终评测方法与效果

评测前在验证集前 300 条样本上比较了 0.35、0.45、0.55、0.65 四个全局阈值：

| 阈值 | 验证集句级准确率 |
|---:|---:|
| 0.35 | 51.6667% |
| 0.45 | 51.6667% |
| 0.55 | 51.6667% |
| 0.65 | 51.6667% |

四个阈值结果相同，按照“准确率优先、其次选择最接近默认 0.5 的阈值”的规则选中 0.45。随后仅用该阈值评估测试集，得到 52.4319% 的句级严格完全匹配率。

进一步诊断显示：

- 单 deal 测试句准确率约 53.26%。
- 测试集中 16 条多 deal 句均未完全匹配。
- 6～7 个非空字段的句子准确率约 81%～83%；字段数量达到 10 个时约 49%，达到 12 个时约 16%。
- 条件准确率较低的复杂字段包括 `send_type`、`net_price`、`send_to`、`buyer_send_type`、`bridge_institution` 和 `send_to_trader`。
- `duplicate_bond_row` 类型的 15 条测试样本本次全部未完全匹配。

## 8. 结果解释与限制

### 8.1 当前结果说明了什么

本次模型已经能够较稳定地识别债券成交中的高频基础字段，字段级 Precision 与 Recall 均约 89%。NER 预热对领域 span 学习有效，结构阶段也在三个 epoch 内明显收敛。

句级指标与字段级指标之间约 37 个百分点的差距，表明继续提升业务效果不能只依赖增加训练轮数。后续重点应放在 deal 数量预测、复杂角色字段、跨字段一致性和多 deal 字段绑定上。

### 8.2 必须披露的数据切分问题

训练后复核发现，当前分组逻辑能够隔离完整 fingerprint，但对共享部分 source fingerprint 的不同 multi-merge 组合没有建立全局连通分量。拆分后出现了跨集合的完全相同文本：

| 集合对 | 完全相同文本数 |
|---|---:|
| Train / Validation | 712 |
| Train / Test | 696 |
| Validation / Test | 229 |

测试集中与训练集存在相同文本的 798 条样本，句级准确率约 56.14%；其余 230 条未见文本准确率约 39.57%。因此 **52.43% 可作为昨晚实验的工程结果，但不能作为严格无泄漏泛化指标**。

后续正式对比实验应先按所有 source fingerprint 的连通分量以及规范化文本哈希进行联合分组，冻结无泄漏测试集，再重新报告模型效果。

## 9. 产物位置

| 产物 | 路径 |
|---|---|
| NER 最佳模型 | `test/outputs/ner/core_4060/best` |
| Structure 最佳模型 | `test/outputs/structure/core_4060/best` |
| NER 训练配置 | `test/outputs/ner/core_4060/training_config.json` |
| NER 训练结果 | `test/outputs/ner/core_4060/training_result.json` |
| Structure 训练配置 | `test/outputs/structure/core_4060/training_config.json` |
| Structure 训练结果 | `test/outputs/structure/core_4060/training_result.json` |
| 最终句级评测 | `test/outputs/structure/core_4060/eval_sentence_acc.json` |
| 数据切分统计 | `test/data/core_4060/split_stats.json` |
| 数据校验报告 | `test/logs/validate_core_4060.json` |
| 完整训练日志 | `test/logs/` |

## 10. 总结

昨晚训练成功跑通了完整的债券领域两阶段训练链路，并取得核心 21 字段 **52.43% 单句严格准确率、89.32% 字段级 Micro-F1**。模型已能较好完成基础字段识别，但经典 span 架构在多 deal 记录绑定、复杂业务角色和整句完整性方面仍有明显提升空间。同时，当前测试集存在同源文本泄漏，下一轮正式实验应首先修复切分，再以无泄漏句级准确率作为模型选择和最终验收指标。
