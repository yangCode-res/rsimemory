# Langfuse 轨迹分析

目标是让每次进化能沿着“实际诊断输入 → 模型提出的理由 → 生成代码 → 训练/验证结果与成本”检查。理由是模型报告的假设，性能变化仍须 replay 验证；不能把模型提出的内存泄漏、缓存收益等直接当成测量结论。

## 新运行

CLI 自动在运行目录写入 `traces.jsonl`。记录器使用标准库，不依赖 Langfuse SDK，不在 ReAct 或 policy 调用中发送网络请求。记录写入失败不会改变答题结果；失败计数进入根轨迹 metadata。需通过独立导入步骤确认完整性。

一轮运行记录父版本、实际送入 Reflexion 的受限诊断输入、Reflexion 和代码生成的实际模型请求及响应、三个 proposal、候选代码和结果。每题独立 trace，带有 policy、fold、检查点、题型与原始运行的关联；工具 search/read、模型调用与评分为子 observation。模型截断也保留服务端返回的原始响应。

新 proposal 使用 `evidence-v2`：`basis` 为 `training_cases`、`parent_code` 或 `exploratory`，并报告 `rationale`、`supporting_case_ids`、`supporting_cluster_labels`、`uncertainty`。案例引用只能指向实际诊断输入中的训练案例；聚类引用只能指向本次返回的标签。代码检查和探索性提案允许空案例引用。这些字段提高可追溯性，不限制 policy 的实现方法。

记录包含任务上下文，留在超算。采集请求正文和模型输出，不记录 HTTP 密钥头。导入器默认只允许 loopback 目标。

## 历史运行 146133

历史导入不重跑模型，不覆盖旧产物，不读取测试集。全量范围包含训练/验证的 4608 条问题记录，以及诊断、三个候选与聚合结果。

旧 Reflexion 先选择四个失败、两个成功案例，但通用列表限长又把实际输入裁成五个。导入包保留六个候选案例供核对；其中实际前五个依据旧实现和保存的计数标记为重建，第六个明确不作为已确认送入的证据。

旧日志未保存完整的逐次模型请求、精确 span 时间和明确 proposal→case/cluster 引用。界面区分 `historical_recorded`、`historical_reconstructed`、`unavailable`。展示时间为导入时间，不能用历史 trace 时间轴计算原始运行延迟。原始已保存的耗时仅作为 metadata 展示。

三个旧提案的缓存、deepcopy、重复存储等描述是模型假设；导入不会补造支持案例或因果联系。候选 2 的生成截断来自已保存错误，缺失的原始生成内容仍标记缺失。候选 3 还改了分词，不能将结果差异单独归因于字符串驻留。

## 导入命令

在超算项目根目录运行；输出目录必须是新的目录。

```sh
.venv/bin/python -m scripts.backfill_traces \
  --run runs/full-v03-local-146133/result \
  --output runs/langfuse-backfill-146133-v2 --scope all

.venv/bin/python -m scripts.export_langfuse \
  --input runs/langfuse-backfill-146133-v2/traces.jsonl \
  --output runs/langfuse-backfill-146133-v2/export-dry-run.json --dry-run
```

正式导入需要环境变量 `LANGFUSE_BASE_URL=http://127.0.0.1:3000`、`LANGFUSE_PUBLIC_KEY` 和 `LANGFUSE_SECRET_KEY`。从部署生成的私有配置读取，避免把密钥写进命令参数或版本库。

```sh
.venv/bin/python -m scripts.export_langfuse \
  --input runs/langfuse-backfill-146133-v2/traces.jsonl \
  --output runs/langfuse-backfill-146133-v2/export-receipt.json
```

新运行可直接导入对应的 `traces.jsonl`，无需 backfill。导入使用 OTLP HTTP JSON 与 v4 ingestion header，按体积分批。回执绑定输入 SHA、服务地址和项目公钥指纹；同一回执跳过已接受批次。服务接受不等于 UI 已完成异步入库，需要另行查询确认。OTLP 不承诺去重，避免删除回执后重复提交。

历史决策根 trace ID 为 `ac9f73e55556abd6462be8f4e27bebea`；完整导航在导入目录的 `index.json`，根 observation 仅放有限导航，避免超大单条请求。

部署与本机 SSH 转发参见 [部署目录](../deploy/langfuse/README.md)。Langfuse 服务和数据库的运行状态、导入回执与浏览器可访问性应分别验证；代码测试通过不能替代部署验收。
