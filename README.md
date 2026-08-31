# GLiNER2 债券成交 76 字段结构化抽取

本仓库基于 [fastino-ai/GLiNER2](https://github.com/fastino-ai/GLiNER2) 开发，目标是从债券成交文本中抽取完整的 deal 结构。项目已将原先只覆盖 21 个高频字段的实验流程升级为完整 **76 个业务字段**的训练、校验和正式评测流程。

> 仓库仅包含代码、字段定义、测试和运行说明，不包含业务数据、模型权重、训练日志或检查点。

## 主要改动

相对于上游 GLiNER2，本仓库增加了以下能力：

- 完整 76 字段 schema，定义位于 [`test/schema/deal_field_descriptions.json`](test/schema/deal_field_descriptions.json)。
- 数据准备阶段的字段契约检查：原始字段和 schema 不一致时直接终止。
- NER 与 Structure 训练前的全字段正例覆盖检查。
- 严格无泄漏切分：共享 source fingerprint、fingerprint stem 或规范化文本的样本会被放入同一连通分量。
- 训练集字段覆盖修正：只移动完整连通分量，确保全部 76 字段在训练集中都有正例。
- 低频和易混淆字段的聚焦训练，包括买卖方向、发送方向、桥机构、双方结算、双方账户、双方交易员及不同收益率类型。
- 三阶段服务器训练：全字段 NER 预热、Structure 聚焦训练、完整 schema 低学习率校准。
- 正式评测默认拒绝 21 字段 core 模式，并输出逐字段 support、precision、recall 和 F1。
- 交互式终端使用单行动态进度条；在 `nohup` 或重定向环境中自动关闭进度条，避免日志刷屏。

旧的 `core_4060` 结果只评估了 21 个字段，不能作为 76 字段正式业务准确率。

## 环境要求

- Python 3.8+
- PyTorch
- Transformers
- PEFT
- NVIDIA GPU 和 CUDA（正式训练推荐）

安装代码及本地训练依赖：

```bash
git clone https://github.com/zye344724-debug/gliner2.git
cd gliner2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[local]"
```

Windows PowerShell 激活虚拟环境：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[local]"
```

## 输入数据格式

训练数据为 UTF-8 编码的 JSONL 文件：每行一个 JSON 对象。最小结构如下：

```json
{
  "input": "240001 1000 买方机构 出给 卖方机构",
  "output": {
    "json_structures": [
      {
        "deal": {
          "bond_code": {"text": "240001", "start": 0, "end": 6},
          "volume": {"text": "1000", "start": 7, "end": 11},
          "buyer": {"text": "买方机构", "start": 12, "end": 16},
          "seller": {"text": "卖方机构", "start": 20, "end": 24}
        }
      }
    ]
  },
  "fingerprint": "稳定且可复现的样本标识",
  "augmentation": {
    "source_fingerprints": ["原始样本标识"]
  }
}
```

数据要求：

- `input` 必须是原始文本。
- `output.json_structures` 是成交列表；单句包含多笔成交时可以有多个 `deal`。
- span 使用 `{text, start, end}`，其中 `start` 包含、`end` 不包含，并且必须满足 `input[start:end] == text`。
- 多值字段使用 span 列表。当前列表字段包括 `send_to`、`send_from`、`contact_info`、`buyer_contact_info` 和 `seller_contact_info`。
- 空字段可以为 `null`；数据准备脚本会统一注入完整 76 字段描述。
- 强烈建议提供稳定的 `fingerprint`。增强或合并样本应在 `augmentation.source_fingerprints` 中保留所有来源，以便执行严格无泄漏切分。
- 全部原始字段必须在 [`deal_field_descriptions.json`](test/schema/deal_field_descriptions.json) 中登记；新增或遗漏字段会触发字段契约错误。

## 传入数据

推荐通过环境变量 `BOND_DATA` 传入数据文件的绝对路径。数据可以位于仓库之外，不需要复制或上传到 GitHub。

Linux/macOS：

```bash
export BOND_DATA=/absolute/path/to/bond_deals.jsonl
```

Windows PowerShell：

```powershell
$env:BOND_DATA = "D:\data\bond_deals.jsonl"
```

也可以只在单次命令中传入：

```bash
BOND_DATA=/absolute/path/to/bond_deals.jsonl bash test/run_full_server.sh
```

## 下载基础模型

训练默认使用 `fastino/gliner2-base-v1`。服务器能够访问 Hugging Face 时，模型会按需下载；也可以提前下载并验证到 `test/models/gliner2-base-v1`：

```bash
python test/ensure_model.py
```

模型目录和所有权重文件已被 `.gitignore` 排除，不会上传到 GitHub。

## 推荐训练流程：Linux GPU 服务器

确认已经设置 `BOND_DATA` 后运行：

```bash
bash test/run_full_server.sh
```

脚本依次执行：

1. 检查 Python、PyTorch、Transformers、CUDA、GPU 和显存。
2. 校验原始字段与完整 76 字段 schema。
3. 拆分可安全拆分的多成交句，并按同源连通分量划分 train/validation/test。
4. 生成完整 schema 主训练集和低频/难字段聚焦训练集。
5. 校验 span、文件格式和训练字段覆盖。
6. 使用平衡数据进行全字段 NER 预热。
7. 使用完整 deal 与聚焦字段族混合数据训练 Structure 模型。
8. 仅使用完整 76 字段 schema，以较低学习率进行最终校准。
9. 在验证集选择阈值，并在测试集执行一次正式评测。

默认训练参数：

| 环境变量 | 默认值 | 说明 |
|---|---:|---|
| `PYTHON_BIN` | `python3` | Python 可执行文件 |
| `VARIANT` | `full_server` | 数据与输出子目录名 |
| `NER_EPOCHS` | `4` | NER 训练轮数 |
| `FOCUS_STRUCTURE_EPOCHS` | `8` | Structure 聚焦训练轮数 |
| `FULL_CALIBRATION_EPOCHS` | `2` | 完整 schema 校准轮数 |
| `BATCH_SIZE` | `4` | 训练 micro batch |
| `EVAL_BATCH_SIZE` | `8` | 评测 batch |
| `GRAD_ACCUM` | `4` | 梯度累积步数 |
| `MAX_LEN` | `384` | 最大文本 word token 数 |
| `NUM_WORKERS` | `4` | DataLoader worker 数 |
| `EVAL_STEPS` | `250` | 验证间隔 |
| `RARE_FIELD_TARGET` | `800` | 低频字段聚焦目标 support |
| `FOCUS_MAX_REPEATS` | `4` | 单样本/字段族最大重复数 |
| `PRECISION` | `bf16` | `bf16`、`fp16` 或 `fp32` |

不支持 BF16 时：

```bash
PRECISION=fp16 bash test/run_full_server.sh
```

显存不足时：

```bash
BATCH_SIZE=2 GRAD_ACCUM=8 PRECISION=fp16 bash test/run_full_server.sh
```

完整服务器方案详见 [`test/FULL_SERVER_TRAINING.md`](test/FULL_SERVER_TRAINING.md)。

## RTX 4060 基线流程

Windows RTX 4060 可以运行：

```powershell
.\test\run_4060_overnight.ps1
```

该脚本同样使用完整 76 字段，但训练轮数和 batch 参数更适合 8GB 显存。详细说明见 [`test/OVERNIGHT_4060.md`](test/OVERNIGHT_4060.md)。

## 分步准备与验证数据

如果希望先检查数据而不训练：

```bash
python test/prepare_data.py \
  --schema-mode full \
  --variant full_server \
  --split-multi \
  --retain-train-multi-max-deals 5 \
  --focus-training \
  --rare-field-target 800 \
  --focus-max-repeats 4

python test/validate_data.py \
  --data-dirs test/data/full_server \
  --out test/logs/full_server/validate_data.json
```

关键生成文件：

```text
test/data/full_server/
├── ner_train_clean.jsonl
├── ner_train_balanced_clean.jsonl
├── structure_train_clean.jsonl
├── structure_train_balanced_clean.jsonl
├── structure_val_clean.jsonl
├── structure_test_clean.jsonl
└── split_stats.json
```

其中 `split_stats.json` 会记录：

- 76 字段原始正例数；
- 严格连通分量数量与最大分量大小；
- train/validation/test 实际规模；
- 每个 split 的逐字段正例数；
- 为保证训练字段覆盖而移动的完整分量；
- 聚焦字段、重复次数和最终训练规模。

## 输出与评测

服务器流程最终模型和评测结果位于：

```text
test/outputs/structure/full_server/
├── best/
├── final/
├── run_meta.json
├── training_result.json
└── eval_sentence_acc.json
```

正式评测指标包括：

- `sentence_exact_match`：一句中的全部 deal 和全部非空字段完全一致才记为正确；
- `field_micro_precision`、`field_micro_recall`、`field_micro_f1`；
- `field_metrics`：76 个字段各自的 support、TP、FP、FN、precision、recall 和 F1；
- `business_full_field_evaluation`：是否按完整 76 字段 schema 评测；
- `business_field_coverage_ready`：测试集是否对全部字段都有非空正例覆盖。

`business_field_coverage_ready=false` 时，不得把总准确率解释成“所有字段均已达到业务标准”。当前数据中部分字段全库仅有 1～4 个正例，聚焦训练只能增加已有样本的优化曝光，不能代替新增真实标注。

## 测试

运行字段契约和训练指标相关测试：

```bash
python -m pip install pytest
python -m pytest -q \
  tests/test_bond_full_field_evaluation.py \
  tests/training/test_metrics.py \
  tests/training/test_trainer_invariants.py \
  tests/training/test_pr15_pr17_runtime.py
```

## 数据与版本管理

以下路径不会进入 Git：

```text
test/data/
test/logs/
test/models/
test/outputs/
```

同时忽略 `*.safetensors`、`*.bin`、`*.pt`、`*.pth`、`*.npy` 和日志文件。提交前建议执行：

```bash
git status --short
git diff --cached --name-only
```

## 上游项目与许可证

核心 GLiNER2 模型、处理器和训练框架来自 [fastino-ai/GLiNER2](https://github.com/fastino-ai/GLiNER2)。本仓库保留上游 Apache License 2.0，详见 [`LICENSE`](LICENSE)。债券76字段数据处理、训练课程、字段契约和评测流程是本仓库面向业务场景增加的扩展。
