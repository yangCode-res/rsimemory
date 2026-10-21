# Langfuse 轨迹分析

目标是让每次进化能沿着“实际诊断输入 → 模型提出的理由 → 生成代码 → 训练/验证结果与成本”检查。理由是模型报告的假设，性能变化仍须 replay 验证；不能把模型提出的内存泄漏、缓存收益等直接当成测量结论。

## 新运行

CLI 自动在运行目录写入 `traces.jsonl`。记录器使用标准库，不依赖 Langfuse SDK，不在 ReAct 或 policy 调用中发送网络请求。记录写入失败不会改变答题结果；失败计数进入根轨迹 metadata。需通过独立导入步骤确认完整性。

一轮运行记录父版本、实际送入 Reflexion 的受限诊断输入、Reflexion 和代码生成的实际模型请求及响应、三个 proposal、候选代码和结果。每题独立 trace，带有 policy、fold、检查点、题型与原始运行的关联；工具 search/read、模型调用与评分为子 observation。模型截断也保留服务端返回的原始响应。

新 proposal 使用 `evidence-v3`：在 `evidence-v2` 的 `basis`、`rationale`、`supporting_case_ids`、`supporting_cluster_labels`、`uncertainty` 之外，增加 `mechanism` 和 `falsification_test`，分别说明操作机制和如何证伪。`basis` 为 `training_cases`、`parent_code` 或 `exploratory`。案例引用只能指向实际诊断输入中的训练案例；聚类引用只能指向本次返回的标签。代码检查和探索性提案允许空案例引用。这些字段提高可追溯性，不限制 policy 的实现方法。`explore-v1` 提示接收父版本活动配置和训练聚合指标，鼓励机制多样的结构性探索；无诊断对照不会收到训练指标、案例或干预结果。

模型 generation 的 metadata 保存每次实际使用的 `max_tokens`、`temperature`、`timeout_s`，可区分反思和代码生成。当前默认分别是 8192/0.7 与 16384/0.4，超时均 900 秒。生成截断单独归档为 `generation_truncated`，原始响应仍可在 generation 的 Output 检查。

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

新运行可直接导入对应的 `traces.jsonl`，无需 backfill。导入使用 OTLP HTTP JSON 与 v4 ingestion header，按体积分批。回执绑定输入 SHA、服务地址和项目公钥指纹；同一回执跳过已接受批次。若服务部分接收批次，该批次记入 `blocked_ids`，不能区分其中具体接受的记录，因此不自动重传，须排查服务端并人工处理。服务接受不等于 UI 已完成异步入库，需要另行查询确认。OTLP 不承诺去重，避免删除回执后重复提交。

历史决策根 trace ID 为 `ac9f73e55556abd6462be8f4e27bebea`；完整导航在导入目录的 `index.json`，根 observation 仅放有限导航，避免超大单条请求。

部署与本机 SSH 转发参见 [部署目录](../deploy/langfuse/README.md)。Langfuse 服务和数据库的运行状态、导入回执与浏览器可访问性应分别验证；代码测试通过不能替代部署验收。

持久存储统一使用 `/file_storage01/home/juanliu/25_ymj/code/memory-policy-rsi/observability/langfuse/`。软件原档、数据库、轨迹对象、配置与密钥都在该目录；`/tmp/rsi-langfuse-1035/` 仅承载可重建的软件缓存及 PID/socket。数据库迁移须停写、逐文件校验，并通过 v4 读取接口核对历史 observation ID。

## 2026-10-21：完整运行接入 v2

修复包括唯一 run/session ID（不再统一为 result）、research/fixture 环境隔离、policy/fold 标签、模型参数、冻结 system prompt 版本与 generation 关联、原生数值 Score。Prompt 名称绑定内容 SHA；注册的只是实际 system messages，不声称完整动态请求已成为 Prompt 模板；完整请求在 generation input。

在管理节点完成运行后执行（不是在无 Langfuse 服务的 GPU 节点执行）：

```sh
.venv/bin/python -m scripts.langfuse_run \
  --run runs/full-v03-local-JOBID/result \
  --output runs/langfuse-import-JOBID-v2 \
  --env-file observability/langfuse/.env
.venv/bin/python -m scripts.verify_langfuse_run \
  --output runs/langfuse-import-JOBID-v2
```

派生轨迹和回执写到新的导入目录，原始 run 不变。验证对照每个 span ID、trace ID、用量、提示词版本、环境和每个 Score ID/值/关联；验证不通过返回非零。HTTP 接受不等于完成。出现缺失须先查队列与 worker 错误，确认数据库无对应记录后才可补交，不清空原回执、不重传整轮。

页面选择 research 环境，并按 session full-v03-local-JOBID、policy/fold 标签筛选。Scores 中 correctness 为逐题原评分，e2e_correctness 仅对 actor_failed 未作答补零；judge 异常不补零。AVG/E2E_AVG 是 replay observation 上的六类等权聚合，不能用 correctness 的全题均值代替。policy_seconds 包含候选隔离开销，policy_peak_bytes 为状态峰值。未配置 GPU 价格时美元面板不代表实际计算免费；看用量和延迟。

memory.ingest span 仅对修复后的新运行生效，不能倒造历史摄入 span。历史实际耗时、缺失请求不得伪造。多层 worker/快照成本细分仍以实际记录能力为界。

部署同时限制 Prisma 每个进程连接池为 8，避免默认连接数随管理节点 CPU 数膨胀；此问题可导致已接受的 span 在关联 Prompt 时被后台丢弃。Podman 共享存储的 readonly incomplete VFS layer 恢复只修改当前账号所属条目的目录权限，由 Podman 自行清理；不删除持久数据库。

151576 实际导入验收：47,198 个唯一 observation、17,320 个原生 Score 全部对账一致，缺失/额外/重复与字段差异均为零；22,397 个 generation 的用量及四类 system prompt 关联一致。第一次后台处理因连接耗尽缺失 7 条，已在 core/full 两表确认不存在后单独补交，审计和恢复回执均保留在导入目录。最终以 verification.json verified=true 为准。

原生实验关联：在 verification.json 为 true 后，执行
`.venv/bin/python -m scripts.link_langfuse_experiments --run runs/full-v03-local-JOBID/result --output runs/langfuse-import-JOBID-v2 --env-file observability/langfuse/.env`。
创建本地 train/validation Dataset items，再用官方 v4 OTel experiment 属性增强已有 question trace/observation（保留原 ID），不重新生成回答，不读取 test，不把历史答案伪作 expected output。Dataset item 注册使用 4 并发且保存逐项回执；实验关联使用单独的 OTLP 增强回执；每个策略/fold 独立运行。导入后逐项核对原生 experiment 与既有 trace/observation 关联。原生逐题均值仍不替代六类等权 AVG。

当前 events_only 部署必须用 /api/public/experiments 与 /api/public/experiment-items 读回核验。旧 dataset-run-items POST 虽接受但不创建新 Experiments 页面记录；早期尝试回执留作审计，不将其算成成功实验关联。v4 实验属性增强先经合成验收，确认原有 span ID、token 与 Score 不变后才用于真实运行；最后再次执行完整入库对账。

151576 v4 实验验收：10 个 native experiment，各训练 772、验证 380 个 item，全部 trace/observation 关联一致。实验属性增强完成后又通过原记录全量对账（47,198 observation，17,320 Score，usage 一致，无重复）。界面 Experiments 可排除 Dataset rsi-integration-synthetic-v2；首页选择 research 环境。外部浏览器不共享应用内浏览器筛选，请在对应浏览器设置。

## 进化优先观测配置

`deploy/langfuse/observability.json` 控制派生总览：enabled 启用、version 绑定不可变视图版本、environment 隔离展示、include_type_metrics 保留六类成绩及百分点变化、include_training_case_links 关联父策略训练案例。调整已发布视图的字段后提升 version，不能覆盖旧视图。原始模型请求和实验归档继续完整保留，不通过这些配置抽样或改变实验。

`scripts.langfuse_run` 导入完成后自动发布 evolution.overview；历史已有导入可以调用 scripts.evolution_overview.publish 派生，不重新导入原记录。总览位于导入目录/version，包含总览根节点、每个候选一个节点、独立 score/trace 回执与 prepare.json；用 scripts.verify_langfuse_run 对这个目录核验。验证按 environment 隔离，不把派生视图当成原始运行的额外执行。

总览展示模型报告的提案、原始代码调用入口、父版本、候选 commit、train/validation 严格 AVG 与 E2E AVG、六类变化、验证成本、相对 BGE 的变化、选择状态和训练案例链接。空严格指标继续为空，不强行补零。支持案例只链接父策略 train，链接存在不证明模型归因成立；hypothesis_verdict 明确为未自动裁定。峰值状态字节与候选隔离开销不可当成纯检索算法成本。

151576：research-overview 中新增 4 条派生观测、18 个聚合指标、6 个已解析训练案例引用，全部入库核验一致；原始 research 记录再次核验不变。没有新增模型调用、读取 test 或重复计算 token。该视图时间表示原执行区间，不是视图生成耗时。进化主链仍须查原始 rsi-evolution-round trace；该总览用于缩短导航，不取代证据与干预。

只生成/恢复总览的独立入口：
```sh
.venv/bin/python -m scripts.evolution_overview --run runs/full-v03-local-JOBID/result --output runs/langfuse-import-JOBID-v2 --env-file observability/langfuse/.env
.venv/bin/python -m scripts.verify_langfuse_run --output runs/langfuse-import-JOBID-v2/evolution-overview-v1
```
命令可重复执行，复用同一回执和同一观测 ID，不增加模型调用；更改视图内容后升级配置 version。
