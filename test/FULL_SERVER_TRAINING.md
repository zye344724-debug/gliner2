# 全 76 字段服务器训练方案

该方案保留全部 76 个业务字段，并针对低频字段和依赖角色/方向语义的难字段增加辅助训练。旧的 `core_4060` 数据与模型不会被覆盖。

## 运行

在 Linux GPU 服务器的仓库根目录执行：

```bash
bash test/run_full_server.sh
```

训练和验证阶段使用单行动态进度条（包含进度、速度、loss 和学习率），不会逐条打印样本。为避免 `tee` 把进度条刷新拆成大量日志行，三个训练阶段直接连接当前终端；模型配置、检查点和最终指标仍会写入各阶段的 `outputs/` 目录。使用 `nohup`、输出重定向或非交互式调度器时，进度条会自动关闭，只保留阶段级日志。

默认参数适合支持 BF16 的单卡服务器：

- NER 全字段聚焦训练：4 epochs
- Structure 全字段 + 聚焦训练：8 epochs
- Structure 完整 schema 低学习率校准：2 epochs
- micro batch：4
- gradient accumulation：4
- 有效 batch：16
- `max_len`：384 个文本 word token
- 低频目标 support：800
- 单样本/字段族最多聚焦重复：4

不支持 BF16 时：

```bash
PRECISION=fp16 bash test/run_full_server.sh
```

显存充足时可以增加 batch；显存不足时降低 micro batch 并增加梯度累积：

```bash
BATCH_SIZE=2 GRAD_ACCUM=8 PRECISION=fp16 bash test/run_full_server.sh
```

## 训练数据组成

`prepare_data.py --focus-training` 会生成：

- `structure_train_clean.jsonl`：每条样本都声明完整 76 字段的主任务数据。
- `structure_train_balanced_clean.jsonl`：完整主任务数据，加难字段族的小 schema 聚焦视图。
- `ner_train_balanced_clean.jsonl`：对应的全字段 NER 预热数据。

聚焦视图按八个业务字段族组织：债券标识、价格收益率、结算、交易双方、人员、发送关系、账户与代码、工作流。买卖方向、发送方向、桥机构、双方结算、双方账户、双方交易员和不同收益率类型即使不属于低频字段，也会进入聚焦训练。

重复次数采用有上限的逆平方根策略。它能提高稀有字段的优化器曝光，又不会把只有 1～4 条的字段扩成大量机械副本。对全库只有极少真实标注的字段，训练脚本会保留并报告，但模型质量仍需要新增真实标注才能可靠保证。

## 三阶段目的

1. NER 阶段学习全部字段的 span 边界。
2. Structure 聚焦阶段同时学习完整 deal 和难字段之间的细粒度区别。
3. Full-schema 校准阶段只使用完整 76 字段主任务，并降低学习率，使最终训练分布与正式推理一致。

验证集和测试集不会加入聚焦重复，正式评测始终使用完整 76 字段。

切分会把共享任一 source fingerprint、fingerprint stem 或规范化文本的样本组成真正的连通分量。当前增强数据存在一个很大的 multi-merge 连通分量，因此严格无泄漏切分不会精确保持 80/10/10；按当前数据和 seed 42，拆分后约为 30,169/1,769/1,779 条 train/validation/test。训练数据更多、验证测试较小是避免同源成交泄漏后的真实结果，不应通过重新拆散连通分量来凑比例。

## 可调参数

脚本支持以下环境变量：

| 参数 | 默认值 |
|---|---:|
| `NER_EPOCHS` | 4 |
| `FOCUS_STRUCTURE_EPOCHS` | 8 |
| `FULL_CALIBRATION_EPOCHS` | 2 |
| `BATCH_SIZE` | 4 |
| `EVAL_BATCH_SIZE` | 8 |
| `GRAD_ACCUM` | 4 |
| `MAX_LEN` | 384 |
| `NUM_WORKERS` | 4 |
| `EVAL_STEPS` | 250 |
| `RARE_FIELD_TARGET` | 800 |
| `FOCUS_MAX_REPEATS` | 4 |
| `PRECISION` | `bf16` |

最终结果位于：

```text
test/outputs/structure/full_server/eval_sentence_acc.json
```

评测文件包含完整字段的句级严格准确率、Micro 指标、逐字段 support/precision/recall/F1，以及测试集是否具备 76 字段正例覆盖的声明。
