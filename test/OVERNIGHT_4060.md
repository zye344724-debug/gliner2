# RTX 4060 首夜训练方案

在仓库根目录的 PowerShell 中运行：

```powershell
.\bond_gliner2_pkg\bond_gliner2_pkg\test\run_4060_overnight.ps1
```

默认使用全部 10000 条原始数据，确保 76 个字段都有非空训练标签。多笔拼接句在可以可靠按标注区间切分时会拆成单笔句，随后按原始 fingerprint 分组做 80/10/10 切分，避免同源样本跨集合泄漏。训练和正式评估均使用完整 76 字段。数据准备会对原始字段与字段描述做严格对账；任何漏字段或新增未登记字段都会直接终止流程，并在统计中列出每个字段的有效标签数。

默认训练参数：NER 1 epoch，structure 3 epochs，micro batch 2，梯度累积 8，FP16，gradient checkpointing，最大长度 256。按当前完整数据和 seed 42 验证生成 26973/3372/3372 条 train/val/test，全部通过 span 与格式校验。

评估会先在验证集 300 条上从 0.35、0.45、0.55、0.65 中选择阈值，再用唯一选中的阈值评估测试集。测试集不会参与调参。

结果文件：`test/outputs/structure/full_4060/eval_sentence_acc.json`。报告除句级严格完全匹配和字段 micro 指标外，还包含 76 字段覆盖契约及逐字段 support、precision、recall、F1。旧的 `core_4060` 结果仅供诊断，不属于正式业务指标。

需要注意：当前 10000 条数据虽然覆盖全部 76 字段，但字段分布极不均衡，部分字段全库只有 1–4 个正例。报告会分别标记“schema 已完整纳入”和“测试集是否具备全部字段的非空正例覆盖”；后者不满足时，`business_field_coverage_ready` 为 `false`，不得将总准确率解释成所有字段均已达到业务标准。此类字段必须补充真实标注，不能用忽略字段或只报 micro 指标掩盖。

如果 8GB 显存仍然 OOM，把两个训练脚本调用中的 `--batch-size 2` 改成 `1`，并把 `--grad-accum 8` 改成 `16`，有效 batch 不变。如果到 06:30 仍未进入评估，可中止训练并从 structure 输出目录的 `best` 检查点评估；脚本的检查点解析会优先选择 `best`。
