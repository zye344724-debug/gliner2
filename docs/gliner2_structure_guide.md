# GLiNER2 项目结构讲解

## 一、项目是什么

GLiNER2 是一个**统一的 schema 驱动信息抽取模型**（205M 参数，基于 DeBERTa 编码器），用**一个模型、一次前向**完成四类任务：

1. **NER 实体抽取** — 给定实体类型（如 `person`、`company`），找出文本中的实体 span
2. **文本分类** — 单标签 / 多标签分类
3. **结构化抽取** — 按 schema 从文本解析出 JSON（实体 + 属性）
4. **关系抽取** — 实体之间的关系三元组

核心思想：**"候选生成 + 打分"** 而不是传统的序列标注。模型把每个"类型/标签"编码成一个 query 向量，与文本的每个 token 做注意力打分，从而判断每个位置是不是某类实体的 start/end。

项目有**两套并存的下游架构**（都共享同一个 DeBERTa 编码器）：

| 架构 | 类 | 状态 | 特点 |
|---|---|---|---|
| **span**（经典） | `SpanExtractor` / `GLiNER2` | 稳定 | 用 `SpanRepLayer` 显式枚举 span 宽度（受 `max_width` 限制），配 LSTM 计数模块 |
| **boundary**（新） | `BoundaryExtractor` | 实验性 | 稀疏的 start/end 配对，支持任意长度 span，没有宽度轴，支持 records（结构化解码）和关系抽取 |

`AutoExtractor.from_pretrained` 会读 checkpoint 的 `architecture` 字段自动选类。

---

## 二、整体分层（学习路线建议按这个顺序）

```
┌─────────────────────────────────────────────────────┐
│  入口层  gliner2/__init__.py, auto.py, engine.py     │  ← 先看这里，知道能调什么
├─────────────────────────────────────────────────────┤
│  推理层  inference/runtime.py, chunking.py,          │  ← 公共 API 如何跑通
│          candidate_decoder.py, span_decoder.py       │
├─────────────────────────────────────────────────────┤
│  任务层  classification/, joint_ie/                  │  ← 三个高级任务如何包装
├─────────────────────────────────────────────────────┤
│  模型层  models/span/, models/boundary/, layers.py   │  ← 核心：两个架构的神经网络
├─────────────────────────────────────────────────────┤
│  处理层  processor.py, processing/, configuration.py │  ← 数据如何变成张量
├─────────────────────────────────────────────────────┤
│  训练层  training/                                   │  ← 训练循环、指标、LoRA
└─────────────────────────────────────────────────────┘
```

---

## 三、根目录与顶层文件

| 文件 | 作用 |
|---|---|
| [README.md](../README.md) | 项目介绍、安装方式、模型列表 |
| [RELEASE.md](../RELEASE.md) | 版本发布记录 |
| [pyproject.toml](../pyproject.toml) | 包配置。注意设计：**基础安装不含 torch**（只有 schema 校验、API 客户端），`gliner2[local]` 才装 torch/transformers 做本地推理 |
| [tutorial/](../tutorial/) | 12 个教程：分类、NER、JSON 抽取、组合 schema、正则校验器、关系抽取、API、训练数据、训练、LoRA、adapter 切换、长文本。**学习 API 用法从这里开始** |
| [docs/boundary_baseline.md](../docs/boundary_baseline.md) | boundary 架构的基线说明 |
| [benchmarks/](../benchmarks/)、[bench/](../bench/) | 性能基准脚本（batching、flash-deberta 等） |
| [scripts/](../scripts/)、[tools/](../tools/) | 开发辅助脚本（检查公开 API 表面、golden 数据捕获等） |
| [tests/](../tests/) | 非常完整的测试套件，约 100 个文件，按模块镜像源码目录组织。`tests/fixtures/` 里有 tiny 模型/checkpoint 用于快速测试 |

## 四、入口层 `gliner2/`

| 文件 | 作用 |
|---|---|
| [__init__.py](../gliner2/__init__.py) | 包入口，导出所有公开 API。**无 torch 也能 import**（有专门的 `test_torch_free_import.py` 保证），torch 相关导入是惰性的 |
| [auto.py](../gliner2/auto.py) | `AutoExtractor` — 根据 checkpoint 的 `architecture` 字段（`span`/`boundary`）自动选择模型类，旧 checkpoint 默认 span |
| [api_client.py](../gliner2/api_client.py) | `GLiNER2API` — 云端 API 客户端，不装 torch 也能用 |
| [configuration.py](../gliner2/configuration.py) | `ExtractorConfig`（两个架构共享的验证配置）+ `BoundaryHeadSettings`（boundary 头的详细超参，如 `start_top_k`、`candidate_budget`、`export_mode` 等） |
| [layers.py](../gliner2/layers.py) | **底层共享神经网络组件**（444 行，必读）：<br>• `create_mlp` — 通用 MLP 工厂<br>• `SpanRepLayer` — span 架构的核心：**markerV0 模式**，用 `[START]`/`[END]` 标记包裹 span 后过注意力池化，得到 span 表示（后面细讲）<br>• `CountLSTM` / `CountLSTMoE` / `CountLSTMv2` — **计数模块**：预测文本中某类实体出现几个（0-20），用于长文本中估计需要抽多少个实体<br>• `CompileSafeGRU` — 手写 GRU（参数名与 `nn.GRU` 一致，checkpoint 兼容，但能被 `torch.compile` 追踪）<br>• `DownscaledTransformer` — 降维 transformer（用于 schema 嵌入的轻量处理） |
| [model.py](../gliner2/model.py) | 兼容性 shim，re-export `SpanExtractorModel`（真正的实现已移到 `models/span/model.py`） |
| [processor.py](../gliner2/processor.py) | `SchemaTransformer` + `PreprocessedBatch` + collate 函数。把原始文本+schema 变成模型的输入张量：tokenize 文本、把 schema 标签拼接成 query 序列、生成训练 targets。含 5 万条 tokenize 缓存 |

## 五、推理层 `gliner2/inference/`（公共 API 的核心）

| 文件 | 作用 |
|---|---|
| [engine.py](../gliner2/inference/engine.py) | 定义公开模型类：`SpanExtractor`、`BoundaryExtractor`、`GLiNER2`（= SpanExtractor 子类，保持历史名字）。两者都混入 `ExtractorRuntimeMixin`，暴露**完全相同的 API** |
| [runtime.py](../gliner2/inference/runtime.py) | **`ExtractorRuntimeMixin`（1379 行，最核心的推理代码）**：公开方法 `extract`、`batch_extract`、`extract_entities`、`classify_text`、长文本分块助手、结果格式化。架构无关——架构差异只在 `_extract_from_batch` 一个钩子里 |
| [schema.py](../gliner2/inference/schema.py) | `Schema`、`RegexValidator`、`StructureBuilder` — 定义你要抽什么：标签、属性（如 person 的 age、name）、嵌套结构（如 building 的 rooms）、正则校验器。**用户写抽取逻辑主要和这个文件打交道** |
| [schema_model.py](../gliner2/inference/schema_model.py) | schema 的数据模型定义（AttributeGroup 等） |
| [chunking.py](../gliner2/inference/chunking.py) | 长文本分块：超长文档切成重叠窗口，跨块合并 span 边界，支持全局 span 处理 |
| [candidate_decoder.py](../gliner2/inference/candidate_decoder.py) | 把 token 级坐标转成字符级 offset（`token_boundaries_to_character_offsets`），boundary 架构用 |
| [span_decoder.py](../gliner2/inference/span_decoder.py) | span 架构的候选解码（把 span rep 分数变成实体列表） |

## 六、任务层

### `gliner2/classification/` — 文本分类

把分类问题形式化为"标签候选的**约束解码**"问题（类似序列标注，但对象是标签）：

| 文件 | 作用 |
|---|---|
| [schema.py](../gliner2/classification/schema.py) | `ClassificationSchema` — 定义标签集、互斥约束（exclusive groups）、必选/可选等 |
| [compiler.py](../gliner2/classification/compiler.py) | 把 schema **编译**成解码问题，按 schema 指纹做 LRU 缓存（相同 schema 不重复编译） |
| [scoring.py](../gliner2/classification/scoring.py) | `ClassificationScorer` — 用模型给每个标签打分 |
| [constraints.py](../gliner2/classification/constraints.py) | 标签之间的约束（互斥、数量限制等，726 行） |
| [decoding/](../gliner2/classification/decoding/) | 三种解码策略：`independent`（独立阈值）、`exact`（精确最优）、`beam`（束搜索） |
| [engine.py](../gliner2/classification/engine.py) | `Classifier` 门面 + `ClassificationConfig`（每次调用的控制参数），`from_pretrained` 只接受加载参数 |
| [result.py](../gliner2/classification/result.py) | 结果构建 |
| [errors.py](../gliner2/classification/errors.py) | 错误类型 |
| [candidates.py](../gliner2/classification/candidates.py) | 候选容器 |
| [long_text.py](../gliner2/classification/long_text.py) | 长文本分类 |

### `gliner2/joint_ie/` — 联合信息抽取（结构化 + 关系）

**在基础模型之上"组合"出来的高层引擎**，不新增模型参数：

| 文件 | 作用 |
|---|---|
| [engine.py](../gliner2/joint_ie/engine.py) | `JointIEConfig` + 高层引擎：把抽取任务变成**候选格（lattice）+ 优化问题** |
| [candidates.py](../gliner2/joint_ie/candidates.py) | 候选生成 |
| [candidate_scores.py](../gliner2/joint_ie/candidate_scores.py) | 候选打分 |
| [lattice.py](../gliner2/joint_ie/lattice.py) | 候选格结构 — 所有可能的实体/角色组合构成搜索空间 |
| [scoring.py](../gliner2/joint_ie/scoring.py) | `RawScorer` + `ScoreLattice` 给候选打分 |
| [optimizers/](../gliner2/joint_ie/optimizers/) | `greedy.py` 贪心、`beam.py` 束搜索 — 在格上找最优解 |
| [constraints.py](../gliner2/joint_ie/constraints.py) | 约束（实体数上限、角色冲突等） |
| [calibration.py](../gliner2/joint_ie/calibration.py) | `Calibrator` 置信度校准 |
| [compiler.py](../gliner2/joint_ie/compiler.py) | schema 编译 |
| [result.py](../gliner2/joint_ie/result.py) | 结果构建 |
| [schema.py](../gliner2/joint_ie/schema.py) | schema 定义 |
| [long_text.py](../gliner2/joint_ie/long_text.py) | 长文本处理 |

## 七、模型层 `gliner2/models/`（最值得学的部分）

### 共享容器

| 文件 | 作用 |
|---|---|
| [base.py](../gliner2/models/base.py) | `BaseExtractorModel` 基类、`EncodedBatch`、`QueryLayout`/`QuerySpec`（query 如何排布：文本+标签的拼接方式） |
| [candidates.py](../gliner2/models/candidates.py) | `ScoredSpanCandidate`、`CandidateSet` — **架构中立的候选容器**，坐标统一为半开区间 `[start, end)` |
| [outputs.py](../gliner2/models/outputs.py) | `CandidateTensorBatch`、`ExtractorOutput` — 模型产出与共享解码器之间的契约。span 架构通过 adapter 填充，boundary 原生产生 |

### `models/span/` — 经典架构（建议先学这个，简单）

| 文件 | 作用 |
|---|---|
| [model.py](../gliner2/models/span/model.py) | `SpanExtractorModel`（933 行）：encoder（DeBERTa）+ `span_rep` + `classifier` + `count_pred` + `count_embed`。前向：编码 → span rep → 计算三类 loss（classification / structure / count） |
| [adapter.py](../gliner2/models/span/adapter.py) | span → 架构中立 `CandidateSet` 的适配器，让 span 模型也能用共享解码器 |

**Span 架构的原理**（对应 GLiNER 原版论文思路）：

- 文本 + 标签列表拼成一条序列，送进 encoder
- 每个标签的 query 向量与 token 向量交互，得到 start/end 概率
- 对候选 span（宽度 ≤ `max_width`，如 12），用 `[START]`/`[END]` marker 包围后过 `SpanRepLayer` 得到 span 表示，再与 query 打分解码
- `CountLSTM` 预测实体数量，辅助长文本

### `models/boundary/` — 新架构（实验性，代码量大但设计清晰）

这是项目最有技术含量的部分（`model.py` 1784 行）。**核心理念：不做宽度轴，直接用 start/end 边界的稀疏配对，任意长度 span 都能抽**。所有坐标是半开区间 `[start, end)`。

| 文件 | 作用 |
|---|---|
| [model.py](../gliner2/models/boundary/model.py) | `BoundaryExtractorModel` + `BoundaryHead`：把 token 状态 + query 状态变成边界边际概率 → 稀疏候选 → 重排对分数 → 加权 loss。刻意与 encoder 解耦，可用合成状态单测过拟合 |
| [heads.py](../gliner2/models/boundary/heads.py) | `BoundaryQueryHead`：**start/end/inside 三个边际头**。全部是点积注意力：投影后的边界状态与 query 状态做 `einsum` 打分再乘 `1/√d`。`inside_prefix` 用前缀和技巧在 O(L) 内算出任意区间内 token 的 inside 分数和（**没有任何 L×L 张量，内存友好**） |
| [proposal.py](../gliner2/models/boundary/proposal.py) | `SparseBoundaryProposer`：从 start/end 边际分数中选出 top-k 位置，稀疏配对生成候选（受 `candidate_budget` 限制），保证候选数可控 |
| [scoring.py](../gliner2/models/boundary/scoring.py) | `SparseBoundaryPairScorer`：对稀疏候选对重打分（pair logits） |
| [pool.py](../gliner2/models/boundary/pool.py) | `DocumentCandidatePool`：文档级候选池，跨 chunk 共享打分 |
| [losses.py](../gliner2/models/boundary/losses.py) | 全套损失：asymmetric focal、balanced BCE、候选对损失、inside 一致性、listwise 重排、abstention、count 损失等 |
| [records.py](../gliner2/models/boundary/records.py) | **结构化 record 解码**（1234 行）：把实体+属性按 schema 组装成 record/event |
| [record_loss.py](../gliner2/models/boundary/record_loss.py) | record 的损失 |
| [record_decode.py](../gliner2/models/boundary/record_decode.py) | record 的解码 |
| [relations.py](../gliner2/models/boundary/relations.py) | 稀疏关系抽取：`TypedRelationPairGenerator` 生成实体对，`SparseRelationScorer` 打分 |
| [encoding.py](../gliner2/models/boundary/encoding.py) | `BoundaryEncoder`：token/boundary 状态编码 |
| [rotary.py](../gliner2/models/boundary/rotary.py) | RoPE 旋转位置编码（用于 endpoint） |
| [content.py](../gliner2/models/boundary/content.py) | span 内容表示 |
| [indexing.py](../gliner2/models/boundary/indexing.py) | 索引工具 |
| [constants.py](../gliner2/models/boundary/constants.py) | 常量（如 `MASK_LOGIT` 有限掩码哨兵值） |
| [targets_device.py](../gliner2/models/boundary/targets_device.py) | target 设备管理 |
| [engine.py](../gliner2/models/boundary/engine.py) | 公开的 `BoundaryExtractor` 类 = 共享 runtime + boundary 模型核心，含区间调度去重叠（`_resolve_flat_spans`） |

## 八、处理与训练层

| 文件 | 作用 |
|---|---|
| [processing/](../gliner2/processing/) | 数据变换细节：<br>• [layouts.py](../gliner2/processing/layouts.py) — query 布局<br>• [targets.py](../gliner2/processing/targets.py) — 训练 targets（`MentionTarget`、`TargetGraph`）<br>• [records.py](../gliner2/processing/records.py) — record 数据处理<br>• [validation.py](../gliner2/processing/validation.py) — 输入校验<br>• [boundary_preprocessing.py](../gliner2/processing/boundary_preprocessing.py) — boundary 预处理 |
| [training/data.py](../gliner2/training/data.py) | `TrainingDataset` — 从 JSONL 构建训练数据 |
| [training/trainer.py](../gliner2/training/trainer.py) | 训练循环（支持梯度累积、分布式） |
| [training/metrics.py](../gliner2/training/metrics.py) | 评估指标 |
| [training/matching.py](../gliner2/training/matching.py) | 预测与 gold 的匹配（span 对齐） |
| [training/sampler.py](../gliner2/training/sampler.py) | 采样器 |
| [training/lora.py](../gliner2/training/lora.py) | LoRA 支持 |

---

## 九、建议的学习路线

1. **先跑起来**：看 [tutorial/2-ner.md](../tutorial/2-ner.md)，用 `GLiNER2.from_pretrained` 抽几个实体，建立直觉
2. **懂 API**：读 `inference/runtime.py` 的公开方法签名 + `inference/schema.py`（Schema 是用户与模型的接口）
3. **懂 span 架构**（简单、经典）：`layers.py` 的 `SpanRepLayer` → `models/span/model.py` 的 forward → `inference/span_decoder.py`
4. **懂数据流**：`processor.py` 看文本+schema 如何变成张量，`models/outputs.py` + `candidates.py` 看模型产出如何变成结果
5. **进阶 boundary 架构**（最有含金量）：`heads.py`（三个边际头 + 前缀和技巧）→ `proposal.py`（稀疏候选）→ `scoring.py`（重排）→ `model.py`（组装 + loss）
6. **任务层**：`classification/`（约束解码思想）→ `joint_ie/`（格 + 束搜索的组合优化思想）
7. **训练**：`training/` + [tutorial/8-train_data.md](../tutorial/8-train_data.md)、[tutorial/9-training.md](../tutorial/9-training.md)
