# RTX 4060 首夜训练方案

在仓库根目录的 PowerShell 中运行：

```powershell
.\bond_gliner2_pkg\bond_gliner2_pkg\test\run_4060_overnight.ps1
```

默认从 10000 条原始数据中覆盖采样 4000 条：先保留全部 76 个字段的稀有正例（低于 16 条的字段全部保留），再用固定随机种子补足预算。因此减少的是冗余样本量，不是业务字段。多笔拼接句在可以可靠按标注区间切分时会拆成单笔句，随后把共享任一 source fingerprint、fingerprint stem 或规范化文本的样本组成连通分量，再进行无泄漏切分。训练和正式评估均声明完整 76 字段；任何字段缺失都会直接终止流程。

默认训练参数：全字段聚焦 NER 1 epoch，难字段聚焦 structure 2 epochs，完整 76 字段低学习率校准 1 epoch；micro batch 2，梯度累积 8，FP16，gradient checkpointing，最大长度 256。按当前数据约为 7331 个优化步，结合昨晚 RTX 4060 实测吞吐、周期验证和最终评测，预计 5～7 小时，中心估计约 6 小时。机器频率、散热和 Windows 后台负载会造成波动。

该脚本保留为 RTX 4060 基线。用于正式服务器长训和难字段聚焦的方案见 `test/FULL_SERVER_TRAINING.md`，入口为 `test/run_full_server.sh`。

评估会先在验证集 200 条上从 0.35、0.45、0.55、0.65 中选择阈值，再用唯一选中的阈值评估测试集。测试集不会参与调参。

结果文件：`test/outputs/structure/full_4060_6h/eval_sentence_acc.json`。报告除句级严格完全匹配和字段 micro 指标外，还包含 76 字段覆盖契约及逐字段 support、precision、recall、F1。旧的 `core_4060` 结果仅供诊断，不属于正式业务指标。

需要注意：当前 10000 条数据虽然覆盖全部 76 字段，但字段分布极不均衡，部分字段全库只有 1–4 个正例。报告会分别标记“schema 已完整纳入”和“测试集是否具备全部字段的非空正例覆盖”；后者不满足时，`business_field_coverage_ready` 为 `false`，不得将总准确率解释成所有字段均已达到业务标准。此类字段必须补充真实标注，不能用忽略字段或只报 micro 指标掩盖。

如果 8GB 显存仍然 OOM，把三个训练脚本调用中的 `--batch-size 2` 改成 `1`，并把 `--grad-accum 8` 改成 `16`，有效 batch 不变。正式训练前先执行 `test/run_4060_smoke.ps1`；它只验证完整 76 字段的三阶段训练和推理链路，不代表模型精度。
